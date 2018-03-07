#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"
echo "Compiling protos in $(pwd)"
cd ../..
protoc avod/protos/*.proto --python_out=.
echo 'Done'