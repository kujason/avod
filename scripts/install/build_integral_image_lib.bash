#!/bin/bash
set -e # exit on first error

build_integral_image_lib()
{
    cd wavedata/wavedata/tools/core/lib
    cmake src
    make
}

build_integral_image_lib
