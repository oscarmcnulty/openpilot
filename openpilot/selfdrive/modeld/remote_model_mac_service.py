#!/usr/bin/env python3
"""Install remote_model_server as a macOS launchd user agent that runs only while a comma is around.

  python -m openpilot.selfdrive.modeld.remote_model_mac_service install [--idle-timeout 300]
  python -m openpilot.selfdrive.modeld.remote_model_mac_service status | uninstall

launchd starts the server when a USB device with the gadget's vendor and product id appears (IOKit
matching), the server holds the Mac awake while the device is connected, and it exits after
--idle-timeout seconds without a device. launchd then starts it again on the next attach.
Logs go to ~/Library/Logs/openpilot-remote-model.log.
"""
import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from openpilot.system.hardware.tici.remote_model_gadget import USB_VID, USB_PID

LABEL = "ai.comma.openpilot.remote-model"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG = Path.home() / "Library" / "Logs" / "openpilot-remote-model.log"
REPO = Path(__file__).resolve().parents[3]


def make_plist(idle_timeout: float, onnx: str | None) -> dict:
  # the venv python, not resolved: following the symlink lands on the base interpreter without the venv packages
  venv_python = REPO / ".venv" / "bin" / "python"
  python = venv_python if venv_python.exists() else Path(sys.executable)
  args = [str(python), "-m", "openpilot.selfdrive.modeld.remote_model_server", "--idle-timeout", str(idle_timeout)]
  if onnx:
    args += ["--onnx", onnx]
  # one entry per provider class: modern macOS publishes IOUSBHostDevice, the legacy IOUSBDevice nub still exists
  usb_match = {f"comma-remote-model-{cls}": {"IOProviderClass": cls, "idVendor": USB_VID, "idProduct": USB_PID, "IOMatchLaunchStream": True}
               for cls in ("IOUSBHostDevice", "IOUSBDevice")}
  plist: dict = {
    "Label": LABEL,
    "ProgramArguments": args,
    "WorkingDirectory": str(REPO),
    "EnvironmentVariables": {
      "PYTHONPATH": str(REPO),
      "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
      "PYTHONUNBUFFERED": "1",
    },
    "StandardOutPath": str(LOG),
    "StandardErrorPath": str(LOG),
    "ProcessType": "Interactive",  # no App Nap or background throttling for the model
    "ThrottleInterval": 5,
    "RunAtLoad": False,
    "KeepAlive": False,  # the server exits itself after idle_timeout, launchd relaunches on the next attach
    "LaunchEvents": {"com.apple.iokit.matching": usb_match},
  }
  return plist


def launchctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
  return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=check)


def domain() -> str:
  return f"gui/{os.getuid()}"


def install(idle_timeout: float, onnx: str | None, dry_run: bool) -> None:
  plist = make_plist(idle_timeout, onnx)
  data = plistlib.dumps(plist)
  if dry_run:
    print(data.decode())
    return
  if sys.platform != "darwin":
    raise SystemExit("launchd agents are macOS only, use --dry-run elsewhere")
  PLIST.parent.mkdir(parents=True, exist_ok=True)
  LOG.parent.mkdir(parents=True, exist_ok=True)
  launchctl("bootout", domain(), str(PLIST))  # replace an existing agent, ignore "not loaded"
  PLIST.write_bytes(data)
  launchctl("bootstrap", domain(), str(PLIST), check=True)
  print(f"installed {PLIST}: starts on USB attach, exits after {idle_timeout:.0f}s without a device, log: {LOG}")


def uninstall() -> None:
  launchctl("bootout", domain(), str(PLIST))
  if PLIST.exists():
    PLIST.unlink()
  print(f"removed {PLIST}")


def status() -> None:
  r = launchctl("print", f"{domain()}/{LABEL}")
  if r.returncode != 0:
    print("not installed")
    return
  for line in r.stdout.splitlines():
    if any(k in line for k in ("state =", "pid =", "last exit", "program =")):
      print(line.strip())
  print(f"log: {LOG}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("action", choices=["install", "uninstall", "status"])
  p.add_argument("--idle-timeout", type=float, default=300, help="seconds without a device before the server exits to standby")
  p.add_argument("--onnx", help="model path passed to the server, default is the big model in the repo")
  p.add_argument("--dry-run", action="store_true", help="print the plist instead of installing")
  args = p.parse_args()
  if args.action == "install":
    install(args.idle_timeout, args.onnx, args.dry_run)
  elif args.action == "uninstall":
    uninstall()
  else:
    status()


if __name__ == "__main__":
  main()
