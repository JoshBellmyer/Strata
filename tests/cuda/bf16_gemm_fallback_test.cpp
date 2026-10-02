// tests/cuda/bf16_gemm_fallback_test.cpp - V100 fork: the prompt path's BF16 GEMM (Gemm::bf16) on this GPU.
//
//   bf16_gemm_fallback_test cublas     cuBLAS's BF16 GemmEx (falls back to the kernel by itself if cuBLAS refuses)
//   bf16_gemm_fallback_test fallback   the FP32 fallback kernel (STRATA_BF16_GEMM_FALLBACK=1) on any card
//
// Random BF16 X [T, K] and W [N, K], Y = X . W^T (+ beta * Y) against an FP64 CPU reference, on shapes like the
// prompt path's (ragged edges included).  Prints the largest relative error and the time; exit 0 when every case
// is within 1e-3 of the largest |Y| (FP32 accumulation of exact BF16 products), 1 otherwise.
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even (no NaNs here)
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "cublas";
#if defined(_WIN32)
    _putenv_s("STRATA_BF16_GEMM_FALLBACK", mode == "fallback" ? "1" : "-1");
#else
    setenv("STRATA_BF16_GEMM_FALLBACK", mode == "fallback" ? "1" : "-1", 1);
#endif
    int dev = 0;
    cudaDeviceProp p{};
    if (!ck(cudaGetDevice(&dev), "cudaGetDevice") || !ck(cudaGetDeviceProperties(&p, dev), "props")) return 1;
    std::printf("bf16_gemm_fallback_test (%s) on %s, sm_%d%d\n", mode.c_str(), p.name, p.major, p.minor);
    cudaStream_t st = nullptr;
    if (!ck(cudaStreamCreate(&st), "stream")) return 1;
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(st, 1 << 20, err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 1; }

    struct Case { int64_t T, N, K; float beta; };
    const Case cases[] = {{1, 1, 16, 0.0f},     {7, 33, 100, 0.0f},     {64, 64, 64, 1.0f},
                          {200, 2560, 640, 0.0f}, {513, 640, 2560, 1.0f}, {512, 2560, 2560, 0.0f}};
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    bool all_ok = true;
    for (const Case& c : cases) {
        std::vector<uint16_t> X((size_t) (c.T * c.K)), W((size_t) (c.N * c.K));
        for (auto& v : X) v = to_bf16(nd(rng));
        for (auto& v : W) v = to_bf16(nd(rng) * 0.05f);
        std::vector<float> Y0((size_t) (c.T * c.N));
        for (auto& v : Y0) v = nd(rng);
        uint16_t *dX = nullptr, *dW = nullptr;
        float* dY = nullptr;
        if (!ck(cudaMalloc(&dX, X.size() * 2), "malloc") || !ck(cudaMalloc(&dW, W.size() * 2), "malloc") ||
            !ck(cudaMalloc(&dY, Y0.size() * 4), "malloc"))
            return 1;
        cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dW, W.data(), W.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dY, Y0.data(), Y0.size() * 4, cudaMemcpyHostToDevice);
        gemm.bf16(dX, dW, dY, c.T, c.N, c.K, 0, c.beta);   // warm-up / the first call decides cuBLAS or not
        cudaMemcpy(dY, Y0.data(), Y0.size() * 4, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(st);
        const auto t0 = std::chrono::steady_clock::now();
        gemm.bf16(dX, dW, dY, c.T, c.N, c.K, 0, c.beta);
        if (!ck(cudaStreamSynchronize(st), "gemm")) return 1;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> Y(Y0.size());
        cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost);
        double max_ref = 0, max_err = 0;
        for (int64_t t = 0; t < c.T; ++t)
            for (int64_t n = 0; n < c.N; ++n) {
                double r = c.beta != 0.0f ? (double) c.beta * Y0[(size_t) (t * c.N + n)] : 0.0;
                for (int64_t k = 0; k < c.K; ++k)
                    r += (double) from_bf16(X[(size_t) (t * c.K + k)]) * (double) from_bf16(W[(size_t) (n * c.K + k)]);
                max_ref = std::max(max_ref, std::fabs(r));
                max_err = std::max(max_err, std::fabs(r - (double) Y[(size_t) (t * c.N + n)]));
            }
        const double rel = max_err / std::max(max_ref, 1e-30);
        const bool ok = rel < 1e-3;
        all_ok = all_ok && ok;
        std::printf("  T=%-5lld N=%-5lld K=%-5lld beta=%.0f  max rel err %.2e  %8.3f ms  %.2f TFLOP/s  %s\n",
                    (long long) c.T, (long long) c.N, (long long) c.K, c.beta, rel, ms,
                    2.0 * c.T * c.N * c.K / (ms * 1e9), ok ? "ok" : "FAIL");
        cudaFree(dX);
        cudaFree(dW);
        cudaFree(dY);
    }
    std::printf("%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}
