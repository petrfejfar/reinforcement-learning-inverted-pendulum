"vendor/ffmpeg/bin/ffmpeg.exe" -framerate 50 -y -i output/state_%%3dms.png -c:v libx264 -r 30 -pix_fmt yuv420p simulation.mp4
