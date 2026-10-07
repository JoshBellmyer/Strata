// V100 fork: how fast this PC's CPU threads can stream expert-sized blocks out of RAM - no GPU.
//
// The decode path computes the experts no GPU holds on the CPU: per verify window it streams a few hundred ~3 MB
// expert blobs from the RAM copy, and the engine's timers put that at ~65 GB/s on the 4070 Super + V100 PC.  This
// measures the ceiling for the same access pattern (whole 3,072,000-byte blocks at random block offsets, read by
// threads pinned to the P-cores' logical processors first) in normal 4 KB pages and, when Windows allows it, in 2 MB
// large pages (the "Lock pages in memory" right), so we know how much room the CPU path has and whether large pages
// would help it (STRATA_COMPLEMENT_LARGE_PAGES=1 in the engine).
//
//   v100_membw [--gib 8] [--seconds 3] [--threads 1,4,6,8,11,12,16,20]
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

namespace {
constexpr size_t kBlock = 3072000;   // a Q4_K gate/up + Q5_1 down expert of UD-Q4_K_XL

void* alloc_pages(size_t bytes, bool large, std::string& note) {
#if defined(_WIN32)
    if (large) {
        HANDLE tok = nullptr;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            TOKEN_PRIVILEGES tp{};
            tp.PrivilegeCount = 1;
            if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                (void) AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
            }
            CloseHandle(tok);
        }
        const SIZE_T lp = GetLargePageMinimum();
        if (lp == 0) { note = "no large-page support"; return nullptr; }
        const SIZE_T lb = (bytes + lp - 1) / lp * lp;
        void* p = VirtualAlloc(nullptr, lb, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (!p) {
            const DWORD e = GetLastError();
            note = "refused, error " + std::to_string(e) +
                   (e == 1314 ? " (the account lacks 'Lock pages in memory')"
                    : e == 1450 ? " (too few free 2 MB pages)" : "");
        }
        return p;
    }
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    if (large) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p == MAP_FAILED) { note = "refused (no hugetlb pages)"; return nullptr; }
        return p;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

void pin_to(int lp) {
#if defined(_WIN32)
    GROUP_AFFINITY ga{};
    ga.Group = (WORD) (lp / 64);
    ga.Mask = (KAFFINITY) 1 << (lp % 64);
    (void) SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr);
#else
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(lp, &s);
    (void) pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#endif
}

uint64_t read_block(const uint8_t* p) {
    const uint64_t* q = (const uint64_t*) p;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    const size_t n = kBlock / 8;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 += q[i]; a1 += q[i + 1]; a2 += q[i + 2]; a3 += q[i + 3];
        a4 += q[i + 4]; a5 += q[i + 5]; a6 += q[i + 6]; a7 += q[i + 7];
    }
    for (; i < n; ++i) a0 += q[i];
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

double run(const uint8_t* buf, size_t blocks, int threads, double seconds) {
    std::atomic<bool> go{false}, stop{false};
    std::atomic<uint64_t> total{0}, sink{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t] {
            // one thread per P-core first (LP 0, 2, ..., 10), then their SMT siblings (1, 3, ..., 11), then the
            // E-cores (12-19) - the 13600K's numbering; the engine's 11 pool workers sit on LP 0-11
            pin_to(t < 6 ? 2 * t : t < 12 ? 2 * (t - 6) + 1 : t);
            std::mt19937_64 rng(1234 + t);
            uint64_t n = 0, s = 0;
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            while (!stop.load(std::memory_order_relaxed)) {
                s += read_block(buf + (rng() % blocks) * kBlock);
                ++n;
            }
            total.fetch_add(n);
            sink.fetch_add(s);
        });
    const auto t0 = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& t : ts) t.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sink.load() == 42) std::printf(" ");
    return (double) total.load() * (double) kBlock / secs / 1e9;
}
}  // namespace

int main(int argc, char** argv) {
    double gib = 8, seconds = 3;
    std::vector<int> threads{1, 4, 6, 8, 11, 12, 16, 20};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--gib") gib = std::atof(argv[i + 1]);
        else if (a == "--seconds") seconds = std::atof(argv[i + 1]);
        else if (a == "--threads") {
            threads.clear();
            for (const char* p = argv[i + 1]; *p;) {
                char* e = nullptr;
                const long v = std::strtol(p, &e, 10);
                if (e == p) { ++p; continue; }
                if (v > 0) threads.push_back((int) v);
                p = e;
            }
        }
    }
    const size_t bytes = (size_t) (gib * 1073741824.0) / kBlock * kBlock;
    const size_t blocks = bytes / kBlock;
    std::printf("v100_membw: %zu blocks of %zu bytes (%.1f GiB), %.1f s per run, threads pinned one per P-core, then SMT siblings, then E-cores\n",
                blocks, kBlock, (double) bytes / 1073741824.0, seconds);
    for (const bool large : {false, true}) {
        std::string note;
        uint8_t* buf = (uint8_t*) alloc_pages(bytes, large, note);
        if (!buf) {
            std::printf("  %s pages: %s\n", large ? "2 MB large" : "4 KB", note.empty() ? "allocation failed" : note.c_str());
            continue;
        }
        for (size_t i = 0; i < bytes; i += 4096) buf[i] = (uint8_t) (i >> 12);   // touch every page
        std::printf("  %s pages:\n", large ? "2 MB large" : "4 KB");
        for (const int t : threads) std::printf("    %2d threads: %6.1f GB/s\n", t, run(buf, blocks, t, seconds));
#if defined(_WIN32)
        VirtualFree(buf, 0, MEM_RELEASE);
#else
        munmap(buf, bytes);
#endif
    }
    return 0;
}
