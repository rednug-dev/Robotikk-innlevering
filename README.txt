TurtleBot marker follower - submission files

marker_follower2.py: main program, version 2026-11-12.4-status.
calibration_pi.npz: camera calibration used by the program.
Keep these two files in the same folder.

Run on the configured Raspberry Pi using the existing robot-env environment.
Requires OpenCV with ArUco, NumPy, Picamera2/libcamera, pyserial and DYNAMIXEL SDK.

Dry run (no motor commands):
~/robot-env/bin/python marker_follower2.py --time 30

Command used in the recorded demonstration (enables motors):
~/robot-env/bin/python marker_follower2.py --drive --video --assume-visible-obstacles --distance 0.60 --speed 0.16 --time 60

Ctrl+C requests shutdown. LiDAR is enabled by default. The assumption flag
changes how missing LiDAR returns are handled. --video is optional.

calibration_images/: the 20 original pictures used for camera calibration.
