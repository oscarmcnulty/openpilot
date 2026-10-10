#!/usr/bin/env python3
"""USB gadget for the remote model link.

Presents the aux USB-C port as a vendor-class USB device with two bulk endpoints (FunctionFS),
so a Mac running remote_model_server can act as the USB host and drive the big model.
Runs as root and holds ep0 open for as long as the gadget should exist: closing ep0 tears the
function down. modeld opens the data endpoints (ep1 IN, ep2 OUT) as the comma user.

The gadget is bound all the time. It is dormant while the controller is in host mode (chestnut or
another USB device on the aux port) and the kernel's type-c negotiation makes the device the
peripheral when a Mac is plugged in.

Stdlib only, no openpilot imports: it re-execs itself under sudo and PYTHONPATH is not passed through.
"""
import os
import pwd
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
OWNER = "comma"

# include/uapi/linux/usb/functionfs.h
FUNCTIONFS_STRINGS_MAGIC = 2
FUNCTIONFS_DESCRIPTORS_MAGIC_V2 = 3
FUNCTIONFS_HAS_FS_DESC, FUNCTIONFS_HAS_HS_DESC, FUNCTIONFS_HAS_SS_DESC = 1, 2, 4


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
  body = struct.pack("<III", 3, 3, 5) + fs + hs + ss
  flags = FUNCTIONFS_HAS_FS_DESC | FUNCTIONFS_HAS_HS_DESC | FUNCTIONFS_HAS_SS_DESC
  return struct.pack("<III", FUNCTIONFS_DESCRIPTORS_MAGIC_V2, 12 + len(body), flags) + body


def strings() -> bytes:
  body = struct.pack("<H", 0x0409) + b"openpilot remote model\0"
  return struct.pack("<IIII", FUNCTIONFS_STRINGS_MAGIC, 16 + len(body), 1, 1) + body


def log(msg: str) -> None:
  print(f"remote_model_gadget: {msg}", flush=True)


def device_serial() -> str:
  """The device serial, stable across boots: macOS remembers accessory approvals by vendor, product and serial."""
  for tok in Path("/proc/cmdline").read_text().split():
    if tok.startswith("androidboot.serialno="):
      return tok.split("=", 1)[1]
  return "unknown"


def setup_configfs(uid: int, gid: int) -> None:
  if not os.path.ismount(CONFIGFS):
    subprocess.check_call(["mount", "-t", "configfs", "none", str(CONFIGFS)])
  GADGET.mkdir(exist_ok=True)
  (GADGET / "idVendor").write_text(f"0x{USB_VID:04x}")
  (GADGET / "idProduct").write_text(f"0x{USB_PID:04x}")
  (GADGET / "bcdDevice").write_text("0x0100")
  (GADGET / "bcdUSB").write_text("0x0320")
  (GADGET / "strings" / "0x409").mkdir(parents=True, exist_ok=True)
  (GADGET / "strings" / "0x409" / "manufacturer").write_text("comma")
  (GADGET / "strings" / "0x409" / "product").write_text("openpilot remote model")
  (GADGET / "strings" / "0x409" / "serialnumber").write_text(device_serial())
  config = GADGET / "configs" / "c.1"
  (config / "strings" / "0x409").mkdir(parents=True, exist_ok=True)
  (config / "strings" / "0x409" / "configuration").write_text("remote model")
  (config / "MaxPower").write_text("100")
  (GADGET / "functions" / FUNCTION).mkdir(exist_ok=True)
  if not (config / FUNCTION).exists():
    (config / FUNCTION).symlink_to(GADGET / "functions" / FUNCTION)
  FFS_MOUNT.mkdir(parents=True, exist_ok=True)
  if not os.path.ismount(FFS_MOUNT):
    subprocess.check_call(["mount", "-t", "functionfs", "-o", f"uid={uid},gid={gid}", "remote_model", str(FFS_MOUNT)])


def main() -> None:
  if os.geteuid() != 0:
    os.execvp("sudo", ["sudo", "-n", sys.executable, os.path.abspath(__file__)])

  udcs = sorted(p.name for p in UDC_CLASS.iterdir()) if UDC_CLASS.exists() else []
  if not udcs or not CONFIGFS.exists():
    log("no UDC or configfs on this device, gadget disabled")
    while True:
      time.sleep(3600)

  owner = pwd.getpwnam(OWNER)
  setup_configfs(owner.pw_uid, owner.pw_gid)
  ep0 = os.open(FFS_MOUNT / "ep0", os.O_RDWR)
  os.write(ep0, descriptors())
  os.write(ep0, strings())
  (GADGET / "UDC").write_text(udcs[0])
  log(f"gadget up on {udcs[0]} as {USB_VID:04x}:{USB_PID:04x}, endpoints {EP_IN_FILE} {EP_OUT_FILE}")
  # ep0 must stay open and be drained of bus events to keep the function alive. on exit the kernel unbinds the
  # gadget itself when ep0 closes.
  while True:
    os.read(ep0, 64)


if __name__ == "__main__":
  main()
