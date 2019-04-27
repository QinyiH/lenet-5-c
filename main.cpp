#include <iostream>
#include "cmake-build-debug/include/data.h"
#include "cmake-build-debug/include/maths.h"
#include "cmake-build-debug/include/lenet.h"

using namespace std;
using namespace cv;

int main() {
    string file_addr = "/home/assassin/dataset/MNIST";
    vector<Mat> train_imageset, test_imageset;
    vector<Mat> train_labelSet, test_labelSet;
    create_database(file_addr, train_imageset, test_imageset, train_labelSet, test_labelSet);
    cout << "train数据集总数：" << train_labelSet.size() << endl;
    cout << "test数据集总数：" << test_labelSet.size() << endl;

    //***********************lenet 初始化***********************************//
    // CNN网络结构设置
    vector<Layer> layers;

    Layer input_layer_1;// 第一层：输入层
    input_layer_1.type = 'i';
    input_layer_1.iChannel = 1;
    input_layer_1.iSizePic[0] = 28;
    input_layer_1.iSizePic[1] = 28;
    layers.push_back(input_layer_1);

    Layer convolutional_layer_2;// 第二层：卷积层
    convolutional_layer_2.type = 'c';
    convolutional_layer_2.iChannel = 2;
    convolutional_layer_2.iSizeKer = 5;
    convolutional_layer_2.padding = 2;
    convolutional_layer_2.activationfunction_type = ReLU;
    layers.push_back(convolutional_layer_2);

    Layer subsampling_layer_3;// 第三层：降采样层
    subsampling_layer_3.type = 's';
    layers.push_back(subsampling_layer_3);

    Layer convolutional_layer_4;// 第四层：卷积层
    convolutional_layer_4.type = 'c';
    convolutional_layer_4.iChannel = 4;
    convolutional_layer_4.iSizeKer = 5;
    convolutional_layer_4.activationfunction_type = ReLU;
    layers.push_back(convolutional_layer_4);

    Layer subsampling_layer_5;// 第五层：降采样层
    subsampling_layer_5.type = 's';
    layers.push_back(subsampling_layer_5);

    Layer fully_connected_layer_6;// 第六层：全连接层
    fully_connected_layer_6.type = 'f';
    fully_connected_layer_6.iChannel = 120;
    fully_connected_layer_6.activationfunction_type = ReLU;
    layers.push_back(fully_connected_layer_6);

    Layer fully_connected_layer_7;// 第七层：全连接层
    fully_connected_layer_7.type = 'f';
    fully_connected_layer_7.iChannel = 84;
    layers.push_back(fully_connected_layer_7);

    Layer fully_connected_layer_8;// 第八层：全连接层（输出层）
    fully_connected_layer_8.type = 'f';
    fully_connected_layer_8.iChannel = 10;
    fully_connected_layer_8.activationfunction_type = SoftMax;
    layers.push_back(fully_connected_layer_8);

    // 定义初始化参数
    double alpha = 2;// 学习率[0.1,3]
    double eta = 0.5f;// 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
    int batchsize = 10;// 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节
    int epochs = 25;// 训练集整体迭代次数
    //down_sample_type down_samp_type = MaxPooling;// 降采样（池化）类型

    // 依据网络结构设置CNN.layers，初始化一个CNN网络
    lenet LeNet(layers, alpha, eta, batchsize, epochs);
    //****************************************************************************************//

    return 0;
}