import os
import subprocess
import sysconfig
import numpy as np

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

python_path = sysconfig.get_paths()['include']
cpppath = [
  "opendbc/can",
  '/usr/lib/include',
  python_path
]

AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=True,
          help='the minimum build. no tests, tools, etc.')

AddOption('--coverage',
          action='store_true',
          help='build with test coverage options')

AddOption('--kaitai',
          action='store_true',
          help='Regenerate kaitai struct parsers')
          
AddOption('--test',
          action='store_true',
          help='build test files')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

AddOption('--ubsan',
          action='store_true',
          help='turn on UBSan')

SHARED = False

lenv = {
  "PATH": os.environ['PATH'] + ":" + Dir(f"#libs/capnpc-java/{arch}/bin").abspath,
  "LD_LIBRARY_PATH": [Dir(f"#third_party/acados/{arch}/lib").abspath],
  "PYTHONPATH": Dir("#").abspath + ":" + Dir("#pyextra/").abspath,

  "ACADOS_SOURCE_DIR": Dir("#third_party/acados/include/acados").abspath,
  "ACADOS_PYTHON_INTERFACE_PATH": Dir("#pyextra/acados_template").abspath,
  "TERA_PATH": Dir("#").abspath + f"/third_party/acados/{arch}/t_renderer",
}

libpath = [
      f"#third_party/acados/{arch}/lib",
      ]

cflags = []
cxxflags = []

rpath = lenv["LD_LIBRARY_PATH"].copy()
rpath += ["/usr/local/lib"]

rpath += [
    Dir("#cereal").abspath,
    Dir("#common").abspath
  ]

if GetOption('asan'):
  ccflags = ["-fsanitize=address", "-fno-omit-frame-pointer"]
  ldflags = ["-fsanitize=address"]
elif GetOption('ubsan'):
  ccflags = ["-fsanitize=undefined"]
  ldflags = ["-fsanitize=undefined"]
else:
  ccflags = []
  ldflags = []

cflags += ['-DSWAGLOG="\\"common/swaglog.h\\""']
cxxflags += ['-DSWAGLOG="\\"common/swaglog.h\\""']

env = Environment(
  ENV=lenv,
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow",
    "-Wno-unknown-warning-option",
    "-Wno-deprecated-register",
    "-Wno-register",
    "-Wno-inconsistent-missing-override",
    "-Wno-c99-designator",
    "-Wno-reorder-init-list",
    "-Wno-error=unused-but-set-variable",
    "-Wno-c++11-narrowing"
  ] + cflags + ccflags,

  CPPPATH=cpppath + [
    "#",
    "#third_party/acados/include",
    "#third_party/acados/include/blasfeo/include",
    "#third_party/acados/include/hpipm/include",
    "#third_party/catch2/include",
    "#third_party/libyuv/include",
    "#third_party/json11",
    "#third_party/linux/include",
    "#third_party/snpe/include",
    "#third_party/mapbox-gl-native-qt/include",
    "#third_party/qrcode",
    "#third_party",
    "#cereal",
    "#opendbc/can",
  ],

  CC='clang',
  CXX='clang++',
  LINKFLAGS=ldflags,

  RPATH=rpath,

  CFLAGS=["-std=gnu11"] + cflags,
  CXXFLAGS=["-std=c++1z"] + cxxflags,
  LIBPATH=libpath + [
    "#cereal",
    "#third_party",
    "#opendbc/can/",
    "#common",
    "#selfdrive/boardd",
  ],

  CYTHONCFILESUFFIX=".cpp",
  COMPILATIONDB_USE_ABSPATH=True,
  REDNOSE_ROOT="#",
  toolpath=["#rednose_repo/site_scons/site_tools"],
  tools=["default", "cython", "compilation_db", "rednose_filter"]
)

# Cython build enviroment
py_include = sysconfig.get_paths()['include']
envCython = env.Clone()
envCython["CPPPATH"] += [py_include, np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]
  envCython["LIBS"] = [os.path.basename(py_include)]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

Export('envCython')

QCOM_REPLAY = False
Export('env', 'arch', 'QCOM_REPLAY', 'SHARED')

SConscript(['common/SConscript'])
Import('_common')
if SHARED:
  common = abspath(common)
else:
  common = [_common, 'json11', 'lmdb']


Export('common')

envCython = env.Clone()
envCython["CPPPATH"] += [np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]

python_libs = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]

  python_libs.append(os.path.basename(python_path))
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

envCython["LIBS"] = python_libs

Export('envCython')

# Build cereal and messaging
SConscript(['cereal/SConscript'])

cereal = [File('#cereal/libcereal.a')]
messaging = [File('#cereal/libmessaging.a')]
visionipc = [File('#cereal/libvisionipc.a')]
messaging_python = [File('#cereal/messaging/messaging_pyx.so')]

Export('cereal', 'messaging', 'messaging_python', 'visionipc')

# Build rednose library and ekf models

rednose_deps = [
  "#selfdrive/locationd/models/constants.py",
  "#selfdrive/locationd/models/gnss_helpers.py",
]

rednose_config = {
  'generated_folder': '#selfdrive/locationd/models/generated',
  'to_build': {
    'gnss': ('#selfdrive/locationd/models/gnss_kf.py', True, [], rednose_deps),
    'live': ('#selfdrive/locationd/models/live_kf.py', True, ['live_kf_constants.h'], rednose_deps),
    'car': ('#selfdrive/locationd/models/car_kf.py', True, [], rednose_deps),
  },
}

if arch != "larch64":
  rednose_config['to_build'].update({
    'loc_4': ('#selfdrive/locationd/models/loc_kf.py', True, [], rednose_deps),
    'pos_computer_4': ('#rednose/helpers/lst_sq_computer.py', False, [], []),
    'pos_computer_5': ('#rednose/helpers/lst_sq_computer.py', False, [], []),
    'feature_handler_5': ('#rednose/helpers/feature_handler.py', False, [], []),
    'lane': ('#xx/pipeline/lib/ekf/lane_kf.py', True, [], rednose_deps),
  })

Export('rednose_config')
SConscript(['rednose/SConscript'])

SConscript(['third_party/SConscript'])

SConscript(['SConscript'])
SConscript(['cereal/SConscript'])
SConscript(['panda/SConscript'])
SConscript(['panda/board/SConscript'])
SConscript(['opendbc/can/SConscript'])

SConscript(['system/proclogd/SConscript'])

SConscript(['common/kalman/SConscript'])
SConscript(['common/transformations/SConscript'])

SConscript(['selfdrive/modeld/SConscript'])

SConscript(['selfdrive/controls/lib/cluster/SConscript'])
SConscript(['selfdrive/controls/lib/lateral_mpc_lib/SConscript'])
SConscript(['selfdrive/controls/lib/longitudinal_mpc_lib/SConscript'])

SConscript(['selfdrive/locationd/SConscript'])
SConscript(['selfdrive/boardd/SConscript'])
SConscript(['selfdrive/loggerd/SConscript'])

if GetOption('test'):
  SConscript('panda/tests/safety/SConscript')
