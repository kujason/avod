#!/bin/bash
set -e # exit on first error

install_wavedata()
{
    echo "Cloning wavedata ..."
    git clone git@github.com:kujason/wavedata.git
    cd wavedata
    sudo /home/travis/virtualenv/python3.5.2/bin/python setup.py install
    cd ../
}

install_protoc()
{
    # Make sure you grab the latest version
    curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
    # Unzip
    unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
    # Move only protoc* to /usr/bin/
    sudo mv protoc3/bin/protoc /usr/bin/protoc
}

#install_wavedata
install_protoc
# install cmake
sudo apt-get install cmake
