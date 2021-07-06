#!/bin/sh

#download and build qulacs
git clone https://github.com/qulacs/qulacs.git
cd qulacs
./script/build_gcc.sh

#create directory
mkdir ../result