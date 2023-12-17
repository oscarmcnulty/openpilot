#!/usr/bin/python3
from openpilot.common.params import Params

p = Params()

p.delete("CalibrationParams")

print("done")
