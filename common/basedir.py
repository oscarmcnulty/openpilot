import os
from pathlib import Path

from openpilot.common.path import flowpilot_root
from openpilot.system.hardware import PC

BASEDIR = flowpilot_root()

if PC:
  PERSIST = os.path.join(str(Path.home()), ".comma", "persist")
else:
  PERSIST = "/persist"
