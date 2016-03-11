#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ffmpeg -loglevel panic -framerate 50 -i $DIR/output/state_%03dms.png -c:v libx264 -r 30 -pix_fmt yuv420p $DIR/simulation.mp4 -y
