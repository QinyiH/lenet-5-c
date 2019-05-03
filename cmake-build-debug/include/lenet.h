//
// Created by assassin on 19-4-27.
//

#ifndef LENET_LENET_H
#define LENET_LENET_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <maths.h>

using namespace std;
using namespace cv;

typedef struct {
    // 当前层的类别
    char type;// 输入层：i；卷积层：c；降采样层：s；全连接层：f

    // 当前层用的激活函数
    activation_function_type activationfunction_type;

    // 当前层的通道数目
    int iChannel;

    // 降采样率
    //int iSample;// 只针对降采样层，其它层此参数无意义

    // 当前层的输入图片大小
    int iSizePic[2];

    // 当前层的卷积核大小[row col]
    int iSizeKer;// 只针对卷积层，其它层此参数无意义

    // 当前层是否填充
    int padding;// 只针对卷积层

    // 当前层的输出（非全连接层输出，全连接层输出为X_vector）
    //TODO：修改成vector<Mat> X，X改成多通道。
    vector<vector<Mat>> X;// 注意是_batchsize幅输入输出同时处理，所以不是2D，而是3D，维度为[_batchsize, iSizePic[0], iSizePic[1]]

    // 前一层通道对当前层通道的卷积核
    vector<vector<Mat>> Ker;// Ker[I][J], I为前一层通道数，J为当前层通道数。只针对卷积层，其它层此参数无意义

    // 前一层通道对当前层通道的卷积核的偏置
    vector<vector<Mat>> Ker_delta;// Ker_delta[I][J], I为前一层通道数，J为当前层通道数。只针对卷积层，其它层此参数无意义

    // 前一层通道对当前层通道的卷积核的梯度
    vector<vector<Mat>> Ker_grad;// Ker_grad[I][J], I为前一层通道数，J为当前层通道数。只针对卷积层，其它层此参数无意义

    // 当前层与上一层的连接权值
    Mat W;// 只针对全连接层，其它层此参数无意义
    // 注意是W[I列][J行],I为当前层全连接输入个数（上一层的数目），J为当前层数目
    // 比如第7层，就是W[84行x120列]

    // ？？？
    Mat W_delta;// 只针对全连接层，其它层此参数无意义
    // 注意是W[I列][J行],I为当前层全连接输入个数（上一层的数目），J为当前层数目

    // 当前层与上一层的连接权值的梯度
    Mat W_grad;// 只针对全连接层，其它层此参数无意义
    // 注意是W[I列][J行],I为当前层全连接输入个数（上一层的数目），J为当前层数目
    // 比如第7层，就是W[84行x120列]

    // 当前层输出通道的加性偏置
    Mat B;

    // 当前层输出通道的加性偏置的更新量
    Mat B_delta;

    // 当前层输出通道的加性偏置的梯度
    Mat B_grad;

    //MaxPooling层的参数，用于存储最大像素点的位置。
    vector<vector<Point>> max_loc_vector;

    // 当前层输出通道的乘性偏置
    //vector<double> Beta;// 只针对下采样层，其它层此参数无意义

    // 当前层输出通道的乘性偏置的更新量
    //vector<double> Beta_delta;// 只针对下采样层，其它层此参数无意义

    // 当前层输出通道的乘性偏置的梯度
    //vector<double> Beta_grad;// 只针对下采样层，其它层此参数无意义

    // 下采样的输入
    vector<vector<Mat>> X_down;// 注意是_batchsize幅输入输出同时处理，所以不是2D，而是3D，维度为[列_batchsize, 行iChannel*iSizePic[0]*iSizePic[1]]
    // 只针对下采样层，其它层此参数无意义

    // 全连接层的输出，或者下一层为全连接层的当前层所有输出图组合成的一个向量
    vector<Mat> X_vector;// 注意是_batchsize幅输入输出同时处理，所以不是一维向量，而是2D，维度为[列_batchsize, 行iChannel]
    // 只针对全连接层的输出，以及下一层为全连接层的非全连接层（比如LeNet的第二个降采样层）。其它层此参数无意义

    // 当前层的灵敏度(残差)，即当前层的输入（特别注意：此输入为紧接着进激活函数的那个输入，即加了乘性和加性偏置后的输入）对误差的偏导
    vector<vector<Mat>> Delta;// 注意是_batchsize幅输入输出同时处理，所以不是2D，而是3D，维度为[_batchsize, iSizePic[0], iSizePic[1]]
    // 注意只针对非全连接层

    // 当前层的灵敏度(残差)，即当前层的输入对误差的偏导
    vector<Mat> Delta_vector;// 虽是一维向量，但是有_batchsize列，所以是二维的。
    // 注意只针对全连接层，以及下层为全连接层的降采样层、输入层、卷积层等有方块图输出的层

} Layer;

class lenet {
public:

    // 初始化lenet网络
    lenet(const vector<Layer> &layers, float alpha, float eta, int batchsize, int epochs);//down_sample_type down_samp_type

    // CNN网络，训练
    void train(const vector<Mat> &train_x, const vector<Mat> &train_y);

    // CNN网络，测试，返回错误率
    double test(const vector<Mat> &test_x, const vector<Mat> &test_y);


    ////////////////////// 非主要函数 /////////////////////////////////////////////

    // 返回迭代次数
    int get_epochs() { return _epochs; }

    // 返回历次迭代的均方误差
    vector<double> get_ERR() { return _ERR; }

private:

    // 依据网络结构设置CNN.layers, 初始化一个CNN网络
    void init();

    // CNN网络，正向计算(批处理算法,核心是convn用法,和输出层批量映射)
    void feed_forward(const vector<Mat> &train_image);

    // CNN网络，反向传播(批处理算法)
    void back_propagation(const vector<Mat> &train_label);

    // CNN网络，卷积层和输出层的权值更新(附加惯性项)
    void update(void);

    vector<Layer> _layers;

    // 学习率[0.1,3]
    double _alpha;

    // 惯性系数[0,0.95], >=1不收敛，==0为不用惯性项
    double _eta;

    // 每次用batchsize个样本计算一个delta调整一次权值，每十个样本做平均进行调节
    int _batchsize;

    // 训练集整体迭代次数
    int _epochs;

    // 激活函数类型
    activation_function_type _activation_func_type;

    // 降采样（池化）类型
    //down_sample_type _down_sample_type;

    // 历次迭代的均方误差
    vector<double> _ERR;

    // 当前轮的当前批次的均方误差
    double _err;

    // 神经网络的输出，就是最后一层网络的输出图
    vector<Mat> _Y;// 注意是_batchsize幅输入输出同时处理，所以是2D，维度为[列_batchsize, 行iChannel（即10）]
};


#endif //LENET_LENET_H
