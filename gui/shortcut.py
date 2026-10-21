"""Add CorridorKey 2.0 to the Start Menu and the desktop, with its icon:

    python -m gui.shortcut

The shortcuts open the desktop UI (`python -m gui`) with the venv's windowless
Python. On Linux this writes an applications-menu entry instead. Run it again
after moving the repo; --remove takes the shortcuts away.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import APP_USER_MODEL_ID

ROOT = Path(__file__).resolve().parents[1]
NAME = "CorridorKey 2.0"
ICONS = Path(__file__).resolve().parent / "icons"

# PowerShell cannot set a shortcut's AppUserModel.ID on its own. With it, the
# pinned shortcut and the running window (which sets the same ID) share one
# taskbar button, and pinning the open window pins this shortcut.
_SET_APP_ID = r"""
using System;
using System.Runtime.InteropServices;
[ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"),
 InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyStore {
    void GetCount(out uint count);
    void GetAt(uint index, out PropertyKey key);
    void GetValue(ref PropertyKey key, out PropVariant value);
    void SetValue(ref PropertyKey key, ref PropVariant value);
    void Commit();
}
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct PropertyKey { public Guid FormatId; public uint PropertyId; }
[StructLayout(LayoutKind.Explicit)]
public struct PropVariant {
    [FieldOffset(0)] public ushort Type;
    [FieldOffset(8)] public IntPtr Value;
}
public static class ShortcutAppId {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHGetPropertyStoreFromParsingName(
        string path, IntPtr bindContext, int flags, ref Guid iid, out IPropertyStore store);
    public static void Set(string shortcut, string appId) {
        Guid iid = typeof(IPropertyStore).GUID;
        IPropertyStore store;
        SHGetPropertyStoreFromParsingName(shortcut, IntPtr.Zero, 2, ref iid, out store);
        var key = new PropertyKey {
            FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), PropertyId = 5 };
        var value = new PropVariant { Type = 31, Value = Marshal.StringToCoTaskMemUni(appId) };
        try { store.SetValue(ref key, ref value); store.Commit(); }
        finally { Marshal.FreeCoTaskMem(value.Value); Marshal.ReleaseComObject(store); }
    }
}
"""


def _ps(value) -> str:
    """A PowerShell single-quoted string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def windows_shortcuts(desktop: bool, remove: bool) -> None:
    pythonw = ROOT / "venv" / "Scripts" / "pythonw.exe"
    if not pythonw.is_file():
        pythonw = Path(sys.executable).with_name("pythonw.exe")
    # WScript.Shell writes .lnk files without extra packages, and resolves the
    # real Start Menu and Desktop folders even when OneDrive redirects them.
    lines = ["$ErrorActionPreference = 'Stop'",
             "$shell = New-Object -ComObject WScript.Shell"]
    if not remove:
        lines.append(f"Add-Type -TypeDefinition {_ps(_SET_APP_ID)}")
    for folder in ["Programs"] + (["Desktop"] if desktop or remove else []):
        lines.append(f"$path = Join-Path $shell.SpecialFolders({_ps(folder)}) {_ps(NAME + '.lnk')}")
        if remove:
            lines.append("if (Test-Path $path) { Remove-Item $path; \"removed $path\" }")
            continue
        lines += [
            "$link = $shell.CreateShortcut($path)",
            f"$link.TargetPath = {_ps(pythonw)}",
            f"$link.Arguments = {_ps('-m gui')}",
            f"$link.WorkingDirectory = {_ps(ROOT)}",
            f"$link.IconLocation = {_ps(str(ICONS / 'corridorkey.ico') + ',0')}",
            f"$link.Description = {_ps('Key green and blue screen footage with CorridorKey 2.0')}",
            "$link.Save()",
            f"[ShortcutAppId]::Set($path, {_ps(APP_USER_MODEL_ID)})",
            "\"wrote $path\"",
        ]
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "\n".join(lines)],
                   check=True)


def linux_entry(remove: bool) -> None:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    path = data_home / "applications" / "corridorkey.desktop"
    if remove:
        path.unlink(missing_ok=True)
        print(f"removed {path}")
        return
    python = ROOT / "venv" / "bin" / "python"
    if not python.is_file():
        python = Path(sys.executable)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={NAME}",
        "Comment=Key green and blue screen footage",
        f'Exec="{python}" -m gui %F',
        f"Path={ROOT}",
        f"Icon={ICONS / 'corridorkey.svg'}",
        "Terminal=false",
        "Categories=AudioVideo;Video;Graphics;",
        "",
    ]))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_desktop", action="store_true",
                    help="Windows: only add the Start Menu entry, no desktop shortcut.")
    ap.add_argument("--remove", action="store_true", help="Remove the shortcuts instead.")
    args = ap.parse_args()

    if sys.platform == "win32":
        windows_shortcuts(desktop=not args.no_desktop, remove=args.remove)
    elif sys.platform.startswith("linux"):
        linux_entry(args.remove)
    else:
        raise SystemExit(f"No shortcuts on {sys.platform}; start the UI with python -m gui.")


if __name__ == "__main__":
    main()
