#!/usr/bin/env sh
    
EXAMPLE=/home/assassin/文档/MNIST  #生成模型训练数据文件夹，即create_imagenet.sh所在文件夹  
#DATA=My_Files/Data_Test             #python脚本处理数据路径，即生成的文件列表.txt文件所在文件夹  
TOOLS=/home/assassin/文档/caffe/build/tools              #caffe的工具库，不用更改  
    
#TRAIN_DATA_ROOT=/home/assassin/文档/MNIST/trainingSet/   #待处理的训练数据  
VAL_DATA_ROOT=/home/assassin/文档/MNIST/testingSet/ 
VAL_DATA=/home/assassin/文档/MNIST/testingSet #待处理的验证数据  
    
#echo "Creating train lmdb..."  
    
rm -rf $EXAMPLE/val_lmdb    #删除已存在的lmdb格式文件，若在已存在lmdb格式的文件夹下再添加lmdb文件，会出现错误  
    
echo "Creating val lmdb..."  
    
GLOG_logtostderr=1 $TOOLS/convert_imageset \  
    --shuffle=ture \  
    $VAL_DATA_ROOT \  
    $VAL_DATA/val.txt \  
    $EXAMPLE/val_lmdb  
        
echo "Done."  