#!/usr/bin/env python3
"""USB gadget for the remote model link.

Presents the aux USB-C port as a vendor-class USB device with two bulk endpoints (FunctionFS),
so a Mac running remote_model_server can act as the USB host and drive the big model.
Runs as root and holds ep0 open for as long as the gadget should exist: closing ep0 tears the
function down. modeld opens the data endpoints (ep1 IN, ep2 OUT) as the comma user.

The gadget is bound all the time. It is dormant while the controller is in host mode (chestnut or
another USB device on the aux port) and the kernel's type-c negotiation decides the role when a Mac
is plugged in. --force-peripheral overrides that for bring-up, and the previous mode is restored on exit.

Stdlib only, no openpilot imports: it re-execs itself under sudo and PYTHONPATH is not passed through.
"""
import argparse
import os
import pwd
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

USB_VID, USB_PID = 0x1209, 0x0001  # pid.codes test PID, private use only
EP_IN, EP_OUT = 0x81, 0x01         # from the host's point of view
FFS_MOUNT = Path("/dev/usb-ffs/remote_model")
EP_IN_FILE, EP_OUT_FILE = FFS_MOUNT / "ep1", FFS_MOUNT / "ep2"  # numbered by order in the descriptors

CONFIGFS = Path("/sys/kernel/config")
GADGET = CONFIGFS / "usb_gadget" / "remote_model"
FUNCTION = "ffs.remote_model"
UDC_CLASS = Path("/sys/class/udc")
CONTROLLER_MODE = Path("/sys/bus/platform/devices/a600000.ssusb/mode")  # dwc3-msm: host, peripheral, none
OWNER = "comma"

# include/uapi/linux/usb/functionfs.h
FUNCTIONFS_STRINGS_MAGIC = 2
FUNCTIONFS_DESCRIPTORS_MAGIC_V2 = 3
FUNCTIONFS_HAS_FS_DESC, FUNCTIONFS_HAS_HS_DESC, FUNCTIONFS_HAS_SS_DESC = 1, 2, 4
EVENT_STRUCT = struct.Struct("<8sB3x")
EVENT_NAMES = ["BIND", "UNBIND", "ENABLE", "DISABLE", "SETUP", "SUSPEND", "RESUME"]


def _iface() -> bytes:
  # bLength, bDescriptorType, bInterfaceNumber, bAlternateSetting, bNumEndpoints, class (vendor), subclass, protocol, iInterface
  return struct.pack("<BBBBBBBBB", 9, 4, 0, 0, 2, 0xFF, 0, 0, 1)


def _ep(addr: int, max_packet: int) -> bytes:
  # bLength, bDescriptorType, bEndpointAddress, bmAttributes (bulk), wMaxPacketSize, bInterval
  return struct.pack("<BBBBHB", 7, 5, addr, 2, max_packet, 0)


def _ss_companion() -> bytes:
  # bLength, bDescriptorType, bMaxBurst, bmAttributes, wBytesPerInterval
  return struct.pack("<BBBBH", 6, 0x30, 0, 0, 0)


def descriptors() -> bytes:
  fs = _iface() + _ep(EP_IN, 64) + _ep(EP_OUT, 64)
  hs = _iface() + _ep(EP_IN, 512) + _ep(EP_OUT, 512)
  ss = _iface() + _ep(EP_IN, 1024) + _ss_companion() + _ep(EP_OUT, 1024) + _ss_companion()
  counts = struct.pack("<III", 3, 3, 5)
  body = counts + fs + hs + ss
  flags = FUNCTIONFS_HAS_FS_DESC | FUNCTIONFS_HAS_HS_DESC | FUNCTIONFS_HAS_SS_DESC
  head = struct.pack("<III", FUNCTIONFS_DESCRIPTORS_MAGIC_V2, 12 + len(body), flags)
  return head + body


def strings() -> bytes:
  body = struct.pack("<H", 0x0409) + b"openpilot remote model\0"
  return struct.pack("<IIII", FUNCTIONFS_STRINGS_MAGIC, 16 + len(body), 1, 1) + body


def write(path: Path, value: str) -> None:
  path.write_text(value)


def log(msg: str) -> None:
  print(f"remote_model_gadget: {msg}", flush=True)


def is_mounted(path: Path) -> bool:
  return any(line.split()[1] == str(path) for line in Path("/proc/mounts").read_text().splitlines())


def setup_configfs(uid: int, gid: int) -> None:
  if not is_mounted(CONFIGFS):
    subprocess.check_call(["mount", "-t", "configfs", "none", str(CONFIGFS)])
  GADGET.mkdir(exist_ok=True)
  write(GADGET / "idVendor", f"0x{USB_VID:04x}")
  write(GADGET / "idProduct", f"0x{USB_PID:04x}")
  write(GADGET / "bcdDevice", "0x0100")
  write(GADGET / "bcdUSB", "0x0320")
  (GADGET / "strings" / "0x409").mkdir(parents=True, exist_ok=True)
  write(GADGET / "strings" / "0x409" / "manufacturer", "comma")
  write(GADGET / "strings" / "0x409" / "product", "openpilot remote model")
  write(GADGET / "strings" / "0x409" / "serialnumber", Path("/proc/sys/kernel/random/boot_id").read_text().strip())
  config = GADGET / "configs" / "c.1"
  (config / "strings" / "0x409").mkdir(parents=True, exist_ok=True)
  write(config / "strings" / "0x409" / "configuration", "remote model")
  write(config / "MaxPower", "100")
  (GADGET / "functions" / FUNCTION).mkdir(exist_ok=True)
  if not (config / FUNCTION).exists():
    (config / FUNCTION).symlink_to(GADGET / "functions" / FUNCTION)
  FFS_MOUNT.mkdir(parents=True, exist_ok=True)
  if not is_mounted(FFS_MOUNT):
    subprocess.check_call(["mount", "-t", "functionfs", "-o", f"uid={uid},gid={gid}", "remote_model", str(FFS_MOUNT)])


def set_controller_mode(mode: str) -> None:
  if CONTROLLER_MODE.exists():
    write(CONTROLLER_MODE, mode)
    log(f"controller mode set to {mode}")
  else:
    log(f"{CONTROLLER_MODE} missing, relying on type-c role negotiation")


def bind_udc() -> str:
  udcs = sorted(p.name for p in UDC_CLASS.iterdir())
  if not udcs:
    raise RuntimeError("no UDC found, kernel gadget support missing?")
  write(GADGET / "UDC", udcs[0])
  return udcs[0]


def teardown(restore_mode: str | None) -> None:
  steps = [lambda: write(GADGET / "UDC", "")]
  if restore_mode is not None:
    steps.append(lambda: set_controller_mode(restore_mode))
  for fn in steps:
    try:
      fn()
    except OSError as e:
      log(f"teardown step failed: {e}")


def serve_ep0(ep0: int) -> None:
  """Logs bus events. Vendor control requests are not used, so any SETUP that reaches us is just acknowledged."""
  while True:
    data = os.read(ep0, EVENT_STRUCT.size * 4)
    for off in range(0, len(data), EVENT_STRUCT.size):
      setup, ev_type = EVENT_STRUCT.unpack_from(data, off)
      name = EVENT_NAMES[ev_type] if ev_type < len(EVENT_NAMES) else str(ev_type)
      log(f"event {name}")
      if name == "SETUP":
        try:
          if setup[0] & 0x80:
            os.write(ep0, b"")
          else:
            os.read(ep0, 0)
        except OSError as e:
          log(f"setup ack failed: {e}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--force-peripheral", action="store_true", help="write 'peripheral' to the dwc3 mode sysfs instead of relying on type-c negotiation")
  args = p.parse_args()

  if os.geteuid() != 0:
    os.execvp("sudo", ["sudo", "-n", sys.executable, os.path.abspath(__file__), *sys.argv[1:]])

  if not UDC_CLASS.exists() or not any(UDC_CLASS.iterdir()) or not CONFIGFS.exists():
    log("no UDC or configfs on this device, gadget disabled")
    while True:
      time.sleep(3600)

  owner = pwd.getpwnam(OWNER)
  setup_configfs(owner.pw_uid, owner.pw_gid)

  ep0 = os.open(FFS_MOUNT / "ep0", os.O_RDWR)
  os.write(ep0, descriptors())
  os.write(ep0, strings())
  udc = bind_udc()
  restore_mode = None
  if args.force_peripheral and CONTROLLER_MODE.exists():
    restore_mode = CONTROLLER_MODE.read_text().strip()
    set_controller_mode("peripheral")
  log(f"gadget up on {udc} as {USB_VID:04x}:{USB_PID:04x}, endpoints {EP_IN_FILE} {EP_OUT_FILE}")

  def on_signal(signum, frame):
    raise SystemExit(0)
  signal.signal(signal.SIGTERM, on_signal)
  signal.signal(signal.SIGINT, on_signal)
  try:
    serve_ep0(ep0)
  finally:
    log("shutting down")
    teardown(restore_mode)
    os.close(ep0)
    time.sleep(0.5)


if __name__ == "__main__":
  main()
