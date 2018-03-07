#!/usr/bin/env bash

cd "$(dirname "$0")"
cd ../..

export PYTHONPATH=$PYTHONPATH:$(pwd)/wavedata
echo $PYTHONPATH

echo "Running unit tests in $(pwd)/avod"
coverage run --source avod -m unittest discover -b --pattern "*_test.py"

#coverage report -m
