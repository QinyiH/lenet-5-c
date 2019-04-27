//
// Created by assassin on 19-4-26.
//

#include "maths.h"

using namespace std;
using namespace cv;

void randperm_array(int serial_num[], int num)
{
    for (int i = 0; i < num; i++)
    {
        serial_num[i] = i;
    }

    int j, temp;

    srand((unsigned)time(NULL));//srand()��������һ���Ե�ǰʱ�俪ʼ���������
    for (int i = num; i > 1; i--)
    {

        j = rand() % i;
        temp = serial_num[i - 1];
        serial_num[i - 1] = serial_num[j];
        serial_num[j] = temp;
    }
}

vector<Mat> activation_function(const vector<Mat> &vector_array, activation_function_type activ_func_type) {
    int page = vector_array.size();
    if (page == 0) {
        cout << "Array3Dd is empty!" << endl << "Array3Dd.activation_function() failed!" << endl;
        vector<Mat> temp;
        return temp;
    }

    switch (activ_func_type) {
        case ReLU: {
            return relu(vector_array);
        }
        default: {
            vector<Mat> temp;
            return temp;
        }
    }
}

//vector<vector<double>> activation_function(const vector<vector<double>> &vector_array, activation_function_type activ_func_type) {
//    int page = vector_array.size();
//    if (page == 0)
//    {
//        cout << "Array2Dd is empty!" << endl << "Array2Dd.activation_function() failed!" << endl;
//        vector<vector<double>> temp;
//        return temp;
//    }
//
//    switch (activ_func_type)
//    {
//        case SoftMax:
//        {
//            return soft_max(vector_array);
//        }
//        case ReLU:
//        {
//            return relu(vector_array);
//        }
//        default:
//        {
//            Array2Dd temp;
//            return temp;
//        }
//    }
//}

vector<Mat> relu(const vector<Mat> &vector_matrix) {
    int nc = vector_matrix.at(0).cols;
    int nl = vector_matrix.at(0).rows;
    if (nc == 0) {
        cout << "Array2Dd is empty!" << endl << "Array3Dd.relu() failed!" << endl;
        vector<Mat> temp;
        return temp;
    }
    vector<Mat> vector_matrix_relu = vector_matrix;
    for (int k = 0; k < vector_matrix.size(); ++k) {
        for (int j = 0; j < nl; j++) {
            uchar *data = vector_matrix_relu.at(k).ptr<uchar>(j);
            for (int i = 0; i < nc; i++) {
                data[i] = data[i] > 0 ? data[i] : 0.0;
            }
        }
    }
    return vector_matrix_relu;
}

vector<int> randperm_vector(int num) {
    vector<int> serial_num;

    for (int i = 0; i < num; i++) {
        serial_num.push_back(i);
    }

    int j, temp;

    srand((unsigned) time(NULL));
    for (int i = num; i > 1; i--) {
        j = rand() % i;
        temp = serial_num.at(i - 1);
        serial_num.at(i - 1) = serial_num.at(j);
        serial_num.at(j) = temp;
    }

    return serial_num;

}

vector<Mat> down_sample_max_pooling(const vector<Mat> &vector_matrix) {
    int nc = vector_matrix.at(0).cols;
    int nl = vector_matrix.at(0).rows;
    int batch = vector_matrix.size();
    if (batch == 0) {
        cout << "FeatureMaps are empty!" << endl << "down_sample_max_pooling() failed!" << endl;
        vector<Mat> temp;
        return temp;
    }
    if (nc == 0) {
        cout << "FeatureMap is empty!" << endl << "down_sample_max_pooling() failed!" << endl;
        vector<Mat> temp;
        return temp;
    }
    vector<Mat> vector_matrix_maxpooling;
    Mat featureMap = Mat::zeros(nl / 2, nc / 2, CV_8UC1);
    vector<Mat> tmp_img_batch = vector_matrix;
    for (int i = 0; i < batch; ++i) {
        vector_matrix_maxpooling.push_back(featureMap);
    }
    for (int k = 0; k < batch; ++k) {
        for (int j = 0; j < nl / 2; j++) {
            uchar *data_in1 = tmp_img_batch.at(k).ptr<uchar>(j * 2);
            uchar *data_in2 = tmp_img_batch.at(k).ptr<uchar>(j * 2 + 1);
            uchar *data_out = vector_matrix_maxpooling.at(k).ptr<uchar>(j);
            for (int i = 0; i < nc / 2; i++) {
                data_out[i] = max(max(data_in1[i * 2], data_in1[i * 2 + 1]), max(data_in2[i * 2], data_in2[i * 2 + 1]));
            }
        }
    }
    return vector_matrix_maxpooling;

}

vector<Mat> reshape2vector(const vector<Mat> &vector_matrix) {
    int batch = vector_matrix.size();
    int dim=pow(vector_matrix.at(0).cols,2);
    vector<Mat> vector_vector;
    Mat tmp;
    for (int i = 0; i < batch; ++i) {
        tmp=vector_matrix.at(i).reshape(0,dim);
        vector_vector.push_back(tmp);
    }
    return vector_vector;
}

vector<double> soft_max(const vector<double> &vector) {
    int dim = vector.size();
    if (dim == 0) {
        cout << "Vector is empty!" << endl << "Vector.soft_max() failed!" << endl;
        std::vector<double> temp;
        return temp;
    }

    std::vector<double> vector_softmax = vector;
    double SUM_Ex = 0;
    for (int j = 0; j < dim; ++j) {
        SUM_Ex = SUM_Ex + exp(vector.at(j));
    }
    for (int i = 0; i < col; i++) {
        // sigmoid function: y = exp(xi) / sum(exp(xi))
        vector_softmax.at(i) = exp(vector.at(i)) / SUM_Ex;
    }
}

return
vector_softmax;
}

//TODO：有空再做,应该是sigmoid
//vector<Mat> soft_max(const vector<Mat> &vector_array) {
//    int page = vector_array.size();
//    if (page == 0)
//    {
//        cout << "Array3Dd is empty!" << endl << "Array3Dd.soft_max() failed!" << endl;
//        vector<Mat> temp;
//        return temp;
//    }
//
//    int col = vector_array.at(0).size();
//    int row = vector_array.at(0).at(0).size();
//
//    Array3Dd vector_array_sigmoid = vector_array;
//
//    for (int i = 0; i < page; i++)
//    {
//        for (int j = 0; j < col; j++)
//        {
//            for (int k = 0; k < row; k++)
//            {
//                // sigmoid function: y = 1 / (1 + exp(-x))
//                double exp_x = exp(- vector_array.at(i).at(j).at(k));
//                vector_array_sigmoid.at(i).at(j).at(k) = 1 / (1 + exp_x);
//            }
//        }
//    }
//
//    return vector_array_sigmoid;
//}

