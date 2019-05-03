//
// Created by assassin on 19-4-26.
//

#ifndef LENET_MATHS_H
#define LENET_MATHS_H

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>

using namespace std;
using namespace cv;

typedef enum {
    SoftMax = 0,
    ReLU,
} activation_function_type;

// 类似matlab中的randperm函数，即得到0~num-1之间的打乱顺序后的数组
void randperm_array(int serial_num[], int num);

//用numeric求和
Mat AverageImage(const vector<Mat> &ImageSet);

void input_layer(vector<Mat> image_set,const Mat avg_image);

//矩阵的激活函数，用于conv
vector<Mat> activation_function(const vector<Mat> &vector_array, activation_function_type activ_func_type);

//向量的激活函数，用于full connected layer
vector<vector<double>>
activation_function(const vector<vector<double>> &vector_array, activation_function_type activ_func_type);

//vector<Mat> soft_max(const vector<Mat> &vector_array);

vector<vector<double>> soft_max(const vector<vector<double>> &vector_array);

vector<Mat> relu(const vector<Mat> &vector_matrix);

//vector<vector<double>> relu(const vector<vector<double>> &vector_array);
//TODO:反向传播 RELU和softmax，以下四个函数
vector<vector<double>> derivation(const vector<vector<double>> &vector_array, activation_function_type activ_func_type);

vector<vector<double>> derivation_soft_max(const vector<vector<double>> &vector_array);

vector<vector<double>> derivation_relu(const vector<vector<double>> &vector_array);

vector<Mat> derivation(const vector<Mat> &vector_array, activation_function_type activ_func_type);

vector<int> randperm_vector(int num);

//TODO down sample & up sample
vector<Mat> down_sample_max_pooling(const vector<Mat> &vector_array);

vector<Mat> up_sample_max_pooling(const vector<Mat> &vector_array3D);

//铺平函数，用于第一层full connected layer
vector<Mat> reshape2vector(const vector<Mat>& vector_matrix);

//full connector layer的计算
// trans:是否转置
vector<Mat> full_connect(const Mat& Weights,const vector<Mat>& vector_vector, const Mat& bias,bool trans);

//计算误差
vector<Mat> calc_error(const vector<Mat>& Y, const vector<Mat>& label);

//批处理矩阵乘法
vector<Mat> matrix_dot(const vector<Mat> &matrix1,const vector<Mat> &matrix2);

//全连接层的偏导
vector<Mat> derivation_fcl(const vector<Mat>& Delta_vector, const vector<Mat> &image_batch);

Mat im2col(const vector<Mat>& FeatureMaps,const int kernelsize);

//相当于径向基函数(RBF)
vector<int> find_max(const vector<Mat> &vector_10);

//计算输出label和label的不同的个数，即预测错误率
double count_dif(const vector<int> &index_1, const vector<int> &index_2);

//批处理单卷积核卷积
vector<Mat> convolution(const vector<Mat> &batch_singlech_inimage, const Mat &kernel,string mode);

void myaccumulate(vector<Mat> &z,vector<Mat> &conv_result);

//计算交叉熵
double calc_cross_entropy(const vector<Mat> output,const vector<Mat> &train_y);

#endif //LENET_MATHS_H
