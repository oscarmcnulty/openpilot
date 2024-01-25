# Dev commands

```sh
# code server startup
ssh OnePlus-7T.lan "sudo login-flowpilot-root code-server"
ssh -N -L 8035:localhost:8035 OnePlus-7T.lan 

# Install to phone over wifi 
adb connect OnePlus-7T.lan
adb install -r android/build/outputs/apk/debug/android-debug.apk

# Open termux remotely
ssh u0_199@OnePlus-7T.lan -p 8022
sudo login-flowpilot-root

# Build and install debug verions of apk
./gradlew android:installDebug 
adb shell am start -n ai.flow.android/ai.flow.android.AndroidLauncher

# Open app
adb shell am start -n ai.flow.android/ai.flow.android.AndroidLauncher

# debug app 
adb shell 
am start -D --user 10 -a android.intent.action.MAIN -n ai.flow.android/ai.flow.android.AndroidLauncher

# Open termux shell through adb
adb shell run-as com.termux files/usr/bin/bash -lic 'export PATH=/data/data/com.termux/files/usr/bin:$PATH; export LD_PRELOAD=/data/data/com.termux/files/usr/lib/libtermux-exec.so; bash -i'

# things missing in setup scripts
tools/install_ubuntu_dependencies.sh

# pulling logs off device
rm /sdcard/openpilot_log.zip
zip -r /sdcard/openpilot_log.zip ~/.comma/media/0/realdata/2024-01-24--03-59-44--*
#on wsl
scp OnePlus-7T.lan:/sdcard/openpilot_log.zip /mnt/c/Users/om/Downloads/openpilot_log.zip
#adb -s 3a516169 pull /sdcard/flowpilot/.flowdrive/media/0/videos/2023-12-18--04-51-31.698.mp4 /Users/om/Downloads/2023-12-18--04-51-31.698.mp4
docker ps # get name of dev container
docker cp /mnt/c/Users/om/Downloads/openpilot_log.zip recursing_yalow:/workspaces/
#on devcontainer
unzip /workspaces/openpilot_log.zip -d /workspaces/
poetry run python tools/plotjuggler/juggle.py /workspaces/root/.comma/media/0/realdata/2024-01-24--02-32-37--0/rlog
./tools/cabana/cabana --no-vipc --data_dir /workspaces/root/.comma/media/0/realdata/ --dbc opendbc/vw_mlb.dbc 2024-01-24--03-59-44

apt install qt5-default
apt install qttools5-dev-tools
apt install ocl-icd-opencl-dev
apt install git-lfs # need to add custom repo
apt install qml-module-qtquick2 \
    qtmultimedia5-dev \
    qtlocation5-dev \
    qtpositioning5-dev \
    qttools5-dev-tools \
    libqt5sql5-sqlite \
    libqt5svg5-dev \
    libqt5charts5-dev \
    libqt5serialbus5-dev  \
    libqt5x11extras5-dev 
```

<img src="https://i.ibb.co/LZtKvfB/Screenshot-from-2022-09-15-22-15-14.png" alt="table" width="1270" />

# What is Flowpilot?

Flowpilot is an open source driver assistance system built on top of openpilot, that can run on most windows/linux and android powered machines. It performs the functions of Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW), Lane Departure Warning (LDW) and Driver Monitoring (DM) for a growing variety of supported car makes, models, and model years maintened by the community.

<table>
  <tr>
    <td><a href="https://youtu.be/L9O-WFmigSA" title="Video By Ender"><img src="https://i3.ytimg.com/vi/L9O-WFmigSA/maxresdefault.jpg"></a></td>
    <td><a href="https://youtu.be/mt86H67DhE0" title="Video By Miso"><img src="https://i3.ytimg.com/vi/mt86H67DhE0/maxresdefault.jpg"></a></td>
    <td><a href="https://youtu.be/06DLmtF6og4" title="Video By Ender"><img src="https://i3.ytimg.com/vi/06DLmtF6og4/maxresdefault.jpg"></a></td>
    <td><a href="https://youtu.be/FBB2XRMej9M" title="Video By Miso"><img src="https://i3.ytimg.com/vi/FBB2XRMej9M/maxresdefault.jpg"></a></td>
  </tr>
</table>

# Running on a Car

For running flowpilot on your car, you need: 

 - A supported machine to run flowpilot i.e. A windows/linux PC or an android phone.
 - A white / grey panda with giraffe or a black/red panda with car harness. 
 - 1x USB-A to USB-A cable for connecting panda to PC and aditionally, an OTG cable is required if connecting panda to phone.
 - One of the [200+ supported cars](https://github.com/commaai/openpilot/blob/master/docs/CARS.md). The community supports Honda, Toyota, Hyundai, Nissan, Kia, Chrysler, Lexus, Acura, Audi, VW, and more. If your car is not supported but has adaptive cruise control and lane-keeping assist, it's likely able to run flowpilot.
 
 For a more detailed overview, see the [wiring and hardware wiki](https://github.com/flowdriveai/flowpilot/wiki/Connecting-to-Car).
 
# Installation:
See the [installation wiki](https://github.com/flowdriveai/flowpilot/wiki/Installation).

# Running With a Virtual Car

It is recommended to develop on a virtual car / simulation before jumping onto testing on a real car. Flowpilot supports CARLA simulation. Optionally, you can use FlowStreamer to test flowpilot with any videogame. For more thorough testing, in addition to simulation, real panda hardware can be put in the loop for a more [thorough testing](https://twitter.com/flowdrive_ai/status/1566680576962478086).

# Community

[<img src="https://assets-global.website-files.com/6257adef93867e50d84d30e2/636e0b5061df29d55a92d945_full_logo_blurple_RGB.svg" width="200">](https://discord.com/invite/APJaQR9nhz)

Flowpilot's core community lives on the official flowdrive [discord server](https://discord.com/invite/APJaQR9nhz). Check the pinned messages or search history through messages to see if your issues or question has been discussed earlier. You may also join [more awesome](https://linktr.ee/flowdrive) openpilot discord communities. 

We also push frequent updates on our [twitter handle](https://twitter.com/flowdrive_ai).

# User Data 

Flowpilot will require your email address for setting up you flowdrive account. Flowpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs. The driver-facing camera is only logged if you explicitly opt-in in settings. The microphone is not recorded.

You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of flowdrive. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to flowdrive for the use of this data.

# Disclaimer 

THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS. NO WARRANTY EXPRESSED OR IMPLIED.
