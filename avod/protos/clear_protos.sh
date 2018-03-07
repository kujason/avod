#!/usr/bin/env bash

set -e

cd "$(dirname "$0")"
echo "Removing old protos from $(dirname "$0")"
find . -name '*_pb2.py'
find . -name '*_pb2.py' -delete
