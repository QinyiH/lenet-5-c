#!/bin/bash
# convert images to lmdb

DATA=/home/assassin/dataset/MNIST
#IMGDIRNAME=trainingSet
#IMGLIST=train.txt
#LMDBNAME=train_lmdb
IMGDIRNAME=testingingSet
IMGLIST=val.txt
LMDBNAME=val_lmdb
rm -rf $DATA/$LMDBNAME
echo 'converting images...'
/home/assassin/文档/caffe/build/tools/convert_imageset --shuffle=true \
$DATA/ $DATA/$IMGLIST $DATA/$LMDBNAME