#!/usr/bin/env python3
"""Install remote_model_server as a macOS launchd user agent that runs only while a comma is around.

  python -m openpilot.selfdrive.modeld.remote_model_mac_service install [--idle-timeout 300]
  python -m openpilot.selfdrive.modeld.remote_model_mac_service uninstall

launchd starts the server when a USB device with the gadget's vendor and product id appears (IOKit
matching), and the server exits after --idle-timeout seconds without a device. launchd then starts it
again on the next attach. Logs go to ~/Library/Logs/openpilot-remote-model.log, and
`launchctl print gui/$UID/ai.comma.openpilot.remote-model` shows the state.
"""
import argparse
import os
import plistlib
import subprocess
from pathlib import Path

from openpilot.system.hardware.tici.remote_model_gadget import USB_VID, USB_PID

LABEL = "ai.comma.openpilot.remote-model"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG = Path.home() / "Library" / "Logs" / "openpilot-remote-model.log"
REPO = Path(__file__).resolve().parents[3]
DOMAIN = f"gui/{os.getuid()}"


def make_plist(idle_timeout: float) -> dict:
  # one match per provider class: modern macOS publishes IOUSBHostDevice, the legacy IOUSBDevice nub still exists
  usb_match = {f"comma-remote-model-{cls}": {"IOProviderClass": cls, "idVendor": USB_VID, "idProduct": USB_PID, "IOMatchLaunchStream": True}
               for cls in ("IOUSBHostDevice", "IOUSBDevice")}
  return {
    "Label": LABEL,
    # the venv python by path: an activated shell is not available to launchd
    "ProgramArguments": [str(REPO / ".venv" / "bin" / "python"), "-m", "openpilot.selfdrive.modeld.remote_model_server",
                         "--idle-timeout", str(idle_timeout)],
    "WorkingDirectory": str(REPO),
    "EnvironmentVariables": {"PYTHONPATH": str(REPO), "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"},
    "StandardOutPath": str(LOG),
    "StandardErrorPath": str(LOG),
    "ProcessType": "Interactive",  # no App Nap or background throttling for the model
    "ThrottleInterval": 5,
    "LaunchEvents": {"com.apple.iokit.matching": usb_match},
  }


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("action", choices=["install", "uninstall"])
  p.add_argument("--idle-timeout", type=float, default=300, help="seconds without a device before the server exits to standby")
  args = p.parse_args()

  subprocess.run(["launchctl", "bootout", DOMAIN, str(PLIST)], capture_output=True)  # ignore "not loaded"
  if args.action == "uninstall":
    PLIST.unlink(missing_ok=True)
    print(f"removed {PLIST}")
    return
  PLIST.parent.mkdir(parents=True, exist_ok=True)
  LOG.parent.mkdir(parents=True, exist_ok=True)
  PLIST.write_bytes(plistlib.dumps(make_plist(args.idle_timeout)))
  subprocess.run(["launchctl", "bootstrap", DOMAIN, str(PLIST)], check=True)
  print(f"installed {PLIST}: starts on USB attach, exits after {args.idle_timeout:.0f}s without a device, log: {LOG}")


if __name__ == "__main__":
  main()
