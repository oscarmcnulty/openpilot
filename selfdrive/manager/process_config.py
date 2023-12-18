import os

from cereal import car
from openpilot.common.params import Params
from openpilot.system.hardware import PC
from openpilot.selfdrive.manager.process import PythonProcess, NativeProcess

WEBCAM = os.getenv("USE_WEBCAM") is not None

def driverview(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started or params.get_bool("IsDriverViewEnabled")

def notcar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and CP.notCar

def iscar(started: bool, params: Params, CP: car.CarParams) -> bool:
  return started and not CP.notCar

def logging(started, params, CP: car.CarParams) -> bool:
  run = (not CP.notCar) or not params.get_bool("DisableLogging")
  return started and run

def ublox_available() -> bool:
  return os.path.exists('/dev/ttyHS0') and not os.path.exists('/persist/comma/use-quectel-gps')

def ublox(started, params, CP: car.CarParams) -> bool:
  use_ublox = ublox_available()
  if use_ublox != params.get_bool("UbloxAvailable"):
    params.put_bool("UbloxAvailable", use_ublox)
  return started and use_ublox

def qcomgps(started, params, CP: car.CarParams) -> bool:
  return started and not ublox_available()

def always_run(started, params, CP: car.CarParams) -> bool:
  return True

def only_onroad(started: bool, params, CP: car.CarParams) -> bool:
  return started

def only_offroad(started, params, CP: car.CarParams) -> bool:
  return not started

def is_f3():
  return Params().get_bool("F3")

  # ai.flow.app:
  #   command: "am start --user 0 -n ai.flow.android/ai.flow.android.AndroidLauncher"
  #   nowait: true
  #   nomonitor: true
  #   platforms: ["android"]

procs = [
  # Athena is interface with athena.comma.ai api
  #DaemonProcess("manage_athenad", "selfdrive.athena.manage_athenad", "AthenadPid"),

  #NativeProcess("camerad", "system/camerad", ["./camerad"], driverview),
  NativeProcess("logcatd", "system/logcatd", ["./logcatd"], only_onroad), # Reads systemd logs and publishes to cereal `androidLog`
  NativeProcess("proclogd", "system/proclogd", ["./proclogd"], only_onroad), # reads processes and CPU usage and publishes to cereal `procLog`
  #PythonProcess("logmessaged", "system.logmessaged", always_run), # takes swaglog messages and publishes them to cereal `logMessage` and `errorLogMessage`
  #PythonProcess("micd", "system.micd", iscar),
#  PythonProcess("timezoned", "system.timezoned", always_run, enabled=not PC),

  #PythonProcess("dmonitoringmodeld", "selfdrive.modeld.dmonitoringmodeld", driverview, enabled=(not PC or WEBCAM)),
#  NativeProcess("encoderd", "system/loggerd", ["./encoderd"], only_onroad),
#  NativeProcess("stream_encoderd", "system/loggerd", ["./encoderd", "--stream"], notcar),
  NativeProcess("loggerd", "system/loggerd", ["./loggerd"], logging), # consumes all cereal sockets and writes them to qlog
  #NativeProcess("modeld", "selfdrive/modeld", ["./modeld"], only_onroad),
  #NativeProcess("mapsd", "selfdrive/navd", ["./mapsd"], only_onroad),
  #PythonProcess("navmodeld", "selfdrive.modeld.navmodeld", only_onroad),
  #NativeProcess("sensord", "system/sensord", ["./sensord"], only_onroad, enabled=not PC),
  #NativeProcess("ui", "selfdrive/ui", ["./ui"], always_run, watchdog_max_dt=(5 if not PC else None)),
  #NativeProcess("soundd", "selfdrive/ui/soundd", ["./soundd"], only_onroad),
  NativeProcess("locationd", "selfdrive/locationd", ["./locationd"], only_onroad),
  NativeProcess("boardd", "selfdrive/boardd", ["./boardd"], always_run, enabled=False),
  PythonProcess("calibrationd", "selfdrive.locationd.calibrationd", only_onroad),
  PythonProcess("torqued", "selfdrive.locationd.torqued", only_onroad),
  PythonProcess("controlsd", "selfdrive.controls.controlsd", only_onroad),
  PythonProcess("deleter", "system.loggerd.deleter", always_run), # cleans up segments when storage hits a minimum level
  #PythonProcess("dmonitoringd", "selfdrive.monitoring.dmonitoringd", driverview, enabled=(not PC or WEBCAM)),
#  PythonProcess("qcomgpsd", "system.qcomgpsd.qcomgpsd", qcomgps, enabled=TICI),
  #PythonProcess("navd", "selfdrive.navd.navd", only_onroad),
  PythonProcess("pandad", "selfdrive.boardd.pandad", always_run),
  PythonProcess("paramsd", "selfdrive.locationd.paramsd", only_onroad),
#  NativeProcess("ubloxd", "system/ubloxd", ["./ubloxd"], ublox, enabled=TICI),
#  PythonProcess("pigeond", "system.sensord.pigeond", ublox, enabled=TICI),
  PythonProcess("plannerd", "selfdrive.controls.plannerd", only_onroad),
  PythonProcess("radard", "selfdrive.controls.radard", only_onroad),
  PythonProcess("thermald", "selfdrive.thermald.thermald", always_run),
  #PythonProcess("tombstoned", "selfdrive.tombstoned", always_run, enabled=not PC),
  #PythonProcess("updated", "selfdrive.updated", only_offroad, enabled=not PC),
  #PythonProcess("uploader", "system.loggerd.uploader", always_run), $ uploads logs and crashes to api.commadotai.com
  PythonProcess("statsd", "selfdrive.statsd", always_run),

  # debug procs
  NativeProcess("bridge", "cereal/messaging", ["./bridge"], notcar),
  #PythonProcess("webjoystick", "tools.bodyteleop.web", notcar),

  #flowpilot processes
  #NativeProcess("modelparsed", "selfdrive/modeld", ["./modelparsed"], only_onroad),
  PythonProcess("keyvald", "selfdrive.keyvald", always_run)
  #NativeProcess("flowpilot", "", ["./gradlew", "desktop:run"], always_run, enabled=PC),
]

managed_processes = {p.name: p for p in procs}
