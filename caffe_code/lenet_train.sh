#!/usr/bin/env sh
set -e

/home/assassin/文档/caffe/build/tools/caffe train --solver=/home/assassin/code/caffe_code/lenet_solver.prototxt $@
~                                                                          