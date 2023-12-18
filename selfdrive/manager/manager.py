#!/usr/bin/env python3
import os
import signal
import time
from pathlib import Path
import subprocess
import sys
import traceback
from typing import List


import psutil
from openpilot.common.params import Params, ParamKeyType
from openpilot.common.basedir import BASEDIR
from openpilot.common import system
from openpilot.common.path import external_android_storage
import cereal.messaging as messaging

from openpilot.selfdrive.boardd.set_time import set_time
from openpilot.selfdrive.manager.filelock import FileLock
from openpilot.selfdrive.manager.helpers import unblock_stdout
from openpilot.selfdrive.manager.process import ensure_running
from openpilot.selfdrive.manager.process_config import managed_processes

from openpilot.system.version import is_dirty, get_commit, get_version, get_origin, get_short_branch, \
                              terms_version, training_version
from openpilot.system.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog, add_file_handler
import openpilot.selfdrive.sentry as sentry

os.chdir(BASEDIR)

POSSIBLE_PNAME_MATRIX = [
    "java",  # linux
    "ai.flow.android",  # android
    "java.exe",  # windows
]
ANDROID_APP = "ai.flow.app"
ENV_VARS = ["USE_GPU", "ZMQ_MESSAGING_PROTOCOL", "ZMQ_MESSAGING_ADDRESS",
            "SIMULATION", "FINGERPRINT", "MSGQ", "PASSIVE"]
UNREGISTERED_DONGLE_ID = "UnregisteredDevice"

def manager_init() -> None:

    set_time(cloudlog)

    params = Params()
    params.clear_all(ParamKeyType.CLEAR_ON_MANAGER_START)

    default_params = [
                    ("CompletedTrainingVersion", "1"),
                    ("DisengageOnAccelerator", "1"),
                    ("HasAcceptedTerms", "1"),
                    ("OpenpilotEnabledToggle", "0"),
                    ("WideCameraOnly", "1"),
                        ]

    if params.get_bool("RecordFrontLock"):
        params.put_bool("RecordFront", True)

    # android specififc
    if system.is_android():
        if os.environ.get("USE_SNPE", None) == "1":
            params.put_bool("UseSNPE", True)
        else:
            params.put_bool("UseSNPE", False)

        # android app cannot access internal termux files, need to copy them over
        # to external storage. rsync is used to copy only modified files.
        internal_assets_dir = os.path.join(BASEDIR, "selfdrive/assets")
        external_android_flowpilot_assets_dir = os.path.join(external_android_storage(), "flowpilot/selfdrive")
        Path(external_android_flowpilot_assets_dir).mkdir(parents=True, exist_ok=True)
        subprocess.check_output(["rsync", "-r", "-u", internal_assets_dir, external_android_flowpilot_assets_dir])

    for k, v in default_params:
        if params.get(k) is None:
            params.put(k, v)
    for k, v in [("CompletedTrainingVersion", "1"), ("HasAcceptedTerms", "1")]:
        params.put(k, v)


    # is this dashcam?
    if os.getenv("PASSIVE") is not None:
        params.put_bool("Passive", bool(int(os.getenv("PASSIVE", "0"))))

    if params.get("Passive") is None:
        raise Exception("Passive must be set to continue")

    # set version params
    params.put("Version", get_version())
    params.put("TermsVersion", terms_version)
    params.put("TrainingVersion", training_version)
    params.put("GitCommit", get_commit(default=""))
    params.put("GitBranch", get_short_branch(default=""))
    params.put("GitRemote", get_origin(default=""))

    if not is_dirty():
        os.environ['CLEAN'] = '1'

    sentry.init(sentry.SentryProject.SELFDRIVE)

    cloudlog.bind_global(dongle_id="", version=get_version(), dirty=is_dirty(), # TODO
                        device="todo")

def manager_prepare() -> None:
  for p in managed_processes.values():
    p.prepare()

def manager_cleanup() -> None:
  # send signals to kill all procs
  for p in managed_processes.values():
    p.stop(block=False)

  # ensure all are killed
  for p in managed_processes.values():
    p.stop(block=True)

  cloudlog.info("everything is dead")



def flowpilot_running():
    params = Params()
    ret = False
    pid_bytes = params.get("FlowpilotPID")
    pid = int.from_bytes(pid_bytes, "little") if pid_bytes is not None else None
    try:
        if pid is not None and psutil.pid_exists(pid):
            p = psutil.Process(pid)
            if p.name() in POSSIBLE_PNAME_MATRIX:
                ret = True
    except Exception:
        pass
    return ret


def append_extras(command: str):
    for var in ENV_VARS:
        val = os.environ.get(var, None)
        if val is not None:
            command += f" -e '{var}' '{val}'"
    return command



def manager_thread() -> None:
    cloudlog.bind(daemon="manager")
    cloudlog.info("manager start")
    cloudlog.info({"environ": os.environ})

    params = Params()

    ignore: List[str] = []
    if params.get("UserID", encoding='utf8') in (None, ):
        ignore.append("uploader")
    if os.getenv("NOBOARD") is not None:
        ignore.append("pandad")
    ignore += [x for x in os.getenv("BLOCK", "").split(",") if len(x) > 0]

    sm = messaging.SubMaster(['deviceState', 'carParams'], poll=['deviceState'])
    pm = messaging.PubMaster(['managerState'])

    ensure_running(managed_processes.values(), False, params=params, CP=sm['carParams'], not_run=ignore)

    started_prev = False

    try:
        while True:
            sm.update()

            started = sm['deviceState'].started

            if started and not started_prev:
                params.clear_all(ParamKeyType.CLEAR_ON_ONROAD_TRANSITION)
            elif not started and started_prev:
                params.clear_all(ParamKeyType.CLEAR_ON_OFFROAD_TRANSITION)

            # initialize and update onroad params, which drives boardd's safety setter thread
            if started != started_prev or sm.frame == 0:
                params.put_bool("IsOnroad", started)
                params.put_bool("IsOffroad", not started)

            started_prev = started

            ensure_running(managed_processes.values(), started, params=params, CP=sm['carParams'], not_run=ignore)

            running_daemons = ["%s%s\u001b[0m" % ("\u001b[32m" if p.proc.is_alive() else "\u001b[31m", p.name)
                    for p in managed_processes.values() if p.proc]

            running_daemons.append("%s%s\u001b[0m" % ("\u001b[32m", "flowinitd"))
            if flowpilot_running():
                running_daemons.append("%s%s\u001b[0m" % ("\u001b[32m", "modeld camerad sensord ui soundd"))

            print(" ".join(running_daemons))
            cloudlog.debug(running_daemons)

            # send managerState
            manager_state_msg = messaging.new_message('managerState')
            manager_state_msg.managerState.processes = [p.get_process_state_msg() for p in managed_processes.values()]
            pm.send('managerState', manager_state_msg)

            # Exit main loop when uninstall/shutdown/reboot is needed
            shutdown = False
            for param in ("DoUninstall", "DoShutdown", "DoReboot"):
                if params.get_bool(param):
                    shutdown = True
                    params.put("LastManagerExitReason", param)
                    cloudlog.warning(f"Shutting down manager - {param} set")

            if shutdown:
                break

            time.sleep(2)

    except Exception:
        print(traceback.format_exc())
    finally:
        cloudlog.info("cleaning up..")
        #params.put_bool("FlowinitReady", False)
        manager_cleanup()

def main():

    with FileLock("flowinit"):
        prepare_only = os.getenv("PREPAREONLY") is not None

        for proc in psutil.process_iter():
            if proc.name() in managed_processes.keys():
                cloudlog.warning(f"{proc.name()} already alive, restarting..")
                proc.kill()

        manager_init()

        # Start UI early so prepare can happen in the background
        #if not prepare_only:
        #    managed_processes['ui'].start()

        manager_prepare()

        if prepare_only:
            return

        # SystemExit on sigterm
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

        try:
            manager_thread()
        except Exception:
            traceback.print_exc()
            sentry.capture_exception()
        finally:
            manager_cleanup()

        params = Params()
        if params.get_bool("DoUninstall"):
            cloudlog.warning("uninstalling")
            HARDWARE.uninstall()
        elif params.get_bool("DoReboot"):
            cloudlog.warning("reboot")
            HARDWARE.reboot()
        elif params.get_bool("DoShutdown"):
            cloudlog.warning("shutdown")
            HARDWARE.shutdown()

if __name__ == "__main__":
    unblock_stdout()

    try:
        # Log all cloud logs to file
        add_file_handler(cloudlog)
        main()
    except Exception:
        #add_file_handler(cloudlog)
        cloudlog.exception("Manager failed to start")

        #try:
        #    managed_processes['ui'].stop()
        #except Exception:
        #    pass

        # Show last 3 lines of traceback
        #error = traceback.format_exc(-3)
        #error = "Manager failed to start\n\n" + error
        #with TextWindow(error) as t:
        #    t.wait_for_exit()

        raise

    # manual exit because we are forked
    sys.exit(0)
