"""V100 fork: give your Windows account the "Lock pages in memory" right (SeLockMemoryPrivilege).

The engine can keep its 42 GB RAM copy of the experts in 2 MB "large pages" (STRATA_COMPLEMENT_LARGE_PAGES=1), which may
let the CPU read them faster. Windows only hands out large pages to accounts with this right. This script adds it for
the account that runs it - the same thing as secpol.msc > Local Policies > User Rights Assignment > "Lock pages in
memory" (which Windows Home does not have). It needs administrator rights (v100\\24a_grant_large_pages.bat asks for
them), and the right takes effect after you sign out and back in (or restart).

    python tools\\v100_grant_lockpages.py            add the right
    python tools\\v100_grant_lockpages.py --remove   take it away again
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes

PRIV = "SeLockMemoryPrivilege"


class LSA_UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", wintypes.LPWSTR)]


class LSA_OBJECT_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", wintypes.HANDLE), ("ObjectName", ctypes.c_void_p),
                ("Attributes", wintypes.ULONG), ("SecurityDescriptor", ctypes.c_void_p),
                ("SecurityQualityOfService", ctypes.c_void_p)]


def user_sid_string() -> tuple[str, str]:
    out = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True).stdout.strip()
    name, sid = [x.strip('"') for x in out.split('","')]
    return name, sid


def main() -> int:
    if sys.platform != "win32":
        print("Windows only")
        return 1
    remove = "--remove" in sys.argv
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("This needs administrator rights: run v100\\24a_grant_large_pages.bat (it asks for them).")
        return 1
    # the account that started the elevated prompt: whoami runs as the same (elevated) user
    name, sid_str = user_sid_string()
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    for f in ("LsaOpenPolicy", "LsaAddAccountRights", "LsaRemoveAccountRights", "LsaClose", "LsaNtStatusToWinError"):
        getattr(advapi, f).restype = ctypes.c_ulong
    advapi.LsaNtStatusToWinError.argtypes = [ctypes.c_ulong]
    sid = ctypes.c_void_p()
    if not advapi.ConvertStringSidToSidW(wintypes.LPCWSTR(sid_str), ctypes.byref(sid)):
        print(f"could not read the account's SID {sid_str}: error {ctypes.get_last_error()}")
        return 1
    attrs = LSA_OBJECT_ATTRIBUTES()
    attrs.Length = ctypes.sizeof(attrs)
    policy = wintypes.HANDLE()
    POLICY_LOOKUP_NAMES, POLICY_CREATE_ACCOUNT = 0x00000800, 0x00000010
    st = advapi.LsaOpenPolicy(None, ctypes.byref(attrs), POLICY_LOOKUP_NAMES | POLICY_CREATE_ACCOUNT,
                              ctypes.byref(policy))
    if st != 0:
        print(f"LsaOpenPolicy failed: Windows error {advapi.LsaNtStatusToWinError(st)}")
        return 1
    buf = ctypes.create_unicode_buffer(PRIV)
    us = LSA_UNICODE_STRING(len(PRIV) * 2, (len(PRIV) + 1) * 2, ctypes.cast(buf, wintypes.LPWSTR))
    if remove:
        st = advapi.LsaRemoveAccountRights(policy, sid, wintypes.BOOLEAN(False), ctypes.byref(us), 1)
    else:
        st = advapi.LsaAddAccountRights(policy, sid, ctypes.byref(us), 1)
    advapi.LsaClose(policy)
    ctypes.windll.kernel32.LocalFree(sid)
    if st != 0:
        print(f"{'removing' if remove else 'adding'} the right failed: Windows error {advapi.LsaNtStatusToWinError(st)}")
        return 1
    print(f"'Lock pages in memory' {'removed from' if remove else 'given to'} {name} ({sid_str}).")
    print("Sign out of Windows and back in (or restart) for it to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
