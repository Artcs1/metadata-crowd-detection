#!/bin/bash

TOTAL=12000
CHUNK=3000

DEVICES=(0 1 2 3)

START=0
i=0

while [ $START -lt $TOTAL ]; do
    NSTART=$((START))
    END=$((START + CHUNK ))
    
    DEVICE=${DEVICES[$((i % ${#DEVICES[@]}))]}
    
    echo "Launching chunk $i: device=$DEVICE, start=$NSTART, end=$END"
    
    python3 process_depth.py --device $DEVICE --start $NSTART --end $END &
    
    START=$((END))
    i=$((i + 1))
done

wait
echo "All tasks completed!"
