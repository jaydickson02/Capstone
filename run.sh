#!/bin/bash

# Run the script
while true
do
    python3 main.py --train --batch_size=32 --episodes=200 --runtime=800 --verbose=1
done