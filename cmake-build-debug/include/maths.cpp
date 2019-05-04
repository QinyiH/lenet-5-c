//
// Created by assassin on 19-4-26.
//

#include "maths.h"

using namespace std;
using namespace cv;

//TODO: 参与点乘的两个Mat矩阵的数据类型（type）只能是 CV_32F、 CV_64FC1、 CV_32FC2、 CV_64FC2 这4种类型中的一种。

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
//TODO: 可能要修改一下精度
Mat AverageImage(const vector<Mat> &ImageSet){
    int Setsize=ImageSet.size();
    if(Setsize==0){
        cout<<"The ImageSet is empty."<<endl;
        Mat tmp;
        return tmp;
    }
    Mat image_avg, image_sum=Mat::zeros(28,28,CV_64FC1);
    for (int i = 0; i < Setsize; i++)
    {
        image_sum=image_sum+ImageSet.at(i);
    }
    image_avg=image_sum/Setsize;
    return image_avg;
}

void input_layer(vector<Mat> &image_set, const Mat avg_image){
    int n=image_set.size();
    for (int i = 0; i < n; i++)
    {
        image_set.at(i)=image_set.at(i)-avg_image;
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

vector<Mat> down_sample_max_pooling(const vector<Mat> &vector_matrix,vector<vector<Point>>& max_loc_vector) {
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
    Mat featureMap = Mat::zeros(nl / 2, nc / 2, CV_32FC1);
    for (int i = 0; i < batch; ++i) {
        vector_matrix_maxpooling.push_back(featureMap);
    }
//    for (int k = 0; k < batch; ++k) {
//        for (int j = 0; j < nl / 2; j++) {
//            uchar *data_in1 = tmp_img_batch.at(k).ptr<uchar>(j * 2);
//            uchar *data_in2 = tmp_img_batch.at(k).ptr<uchar>(j * 2 + 1);
//            uchar *data_out = vector_matrix_maxpooling.at(k).ptr<uchar>(j);
//            for (int i = 0; i < nc / 2; i++) {
//                data_out[i] = max(max(data_in1[i * 2], data_in1[i * 2 + 1]), max(data_in2[i * 2], data_in2[i * 2 + 1]));
//            }
//        }
//    }
    double minVal,maxVal;
    Point minLoc,maxLoc;
    Mat slice_tmp;
    for (int k = 0; k < batch; ++k) {
        for (int i = 0; i < nc/2; ++i) {
            for (int j = 0; j < nl/2; ++j) {
                slice_tmp=vector_matrix.at(k)(Rect(j*2,i*2,2,2));
                minMaxLoc(slice_tmp,&minVal,&maxVal,&minLoc,&maxLoc);
                vector_matrix_maxpooling.at(k).at<float>(j,i)=maxVal;
                max_loc_vector.at(k).push_back(maxLoc);
            }
        }
    }

    return vector_matrix_maxpooling;

}

Mat reshape2vector(const vector<Mat> &vector_matrix) {
    int channel = vector_matrix.size();
    int dim=pow(vector_matrix.at(0).cols,2);
    Mat vector_out;
    Mat tmp1,tmp2;
    for (int i = 0; i < channel; ++i) {
        tmp1=vector_matrix.at(i).reshape(0,dim);
        vconcat(tmp1,tmp2,vector_out);
        tmp1=vector_out;
    }
    return vector_out;
}

vector<vector<Mat>> reshape2channel(const vector<Mat> Delta_vector, const int iChannel, const int SizePic_col, const int SizePic_row){
    int batch=Delta_vector.size();
    for (int i = 0; i < batch; ++i) {

    }
}

vector<Mat> calc_error(const vector<Mat> &Y, const vector<Mat> &label) {
    int batch=Y.size();
    if(batch!=label.size()){
        cout << "bacth数量不一致，train fail!"<<endl;
        vector<Mat> tmp;
        return tmp;
    }
    vector<Mat> batch_error;
    for (int i = 0; i < batch; ++i) {
        batch_error.push_back(Y.at(i)-label.at(i));
    }
    return batch_error;
}

vector<Mat> full_connect(const Mat &Weights, const vector<Mat> &vector_vector, const Mat &bias,bool trans) {
    vector<Mat> vector_a,vector_y;
    int batch=vector_vector.size();
    if(trans){
        for (int i = 0; i < batch; ++i) {
            vector_a.push_back(vector_vector.at(i).t());
        }
    }
    else vector_a=vector_vector;
    for (int j = 0; j < batch; ++j) {
        vector_y.push_back(Weights*vector_a.at(j)+bias);
    }
}

vector<int> find_max(const vector<Mat> &vector_10) {
    int batch=vector_10.size();
    if(batch==0){
        cout<<"There is nothing in the index!"<<endl;
        cout<<"Is the set empty?"<<endl;
        vector<int> tmp;
        return tmp;

    }
    vector<int> vector_index;
    //vector<double>::iterator maximum_loc;
    double max, min;
    Point min_loc, max_loc;
    for (int i = 0; i < batch; ++i) {
        //maximum_loc = std::max_element(std::begin(vector_10), std::end(vector_10));
        minMaxLoc(vector_10.at(i), &min, &max, &min_loc, &max_loc);
        if(max_loc.y>=max_loc.x)
            vector_index.push_back(max_loc.y);
        else
            vector_index.push_back(max_loc.x);
    }
    return vector_index;
}

double count_dif(const vector<int> &index_1, const vector<int> &index_2) {
    int batch=index_1.size();
    int batch_check=index_2.size();
    if (batch!=batch_check){
        cout<<"Test() error: Sizes are not equal!"<<endl;
        double tmp;
        return tmp;
    }
    double count=0;
    for (int i = 0; i < batch; ++i) {
        if (index_1.at(i)!=index_2.at(i)){
            count++;
        }
    }
    return count;

}

vector<Mat> convolution(const vector<Mat> &batch_singlech_inimage, const Mat &kernel, string mode) {
    int batch=batch_singlech_inimage.size();
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < ; ++j) {

        }
    }
}



Mat im2col(const vector<Mat> &FeatureMaps, const int kernelsize) {
    //int Blocksize = kernelsize;
    //int colBlock = kernelsize;
    int d = FeatureMaps.size();
    int m = FeatureMaps.at(0).rows;
    //int n = FeatureMaps.cols;
    int y = m - kernelsize + 1;
    int x = y;
    cv::Mat col = cv::Mat::zeros(x*y,pow(kernelsize,2)*d,CV_32FC1);
    for (int i = 0; i < y; ++i) {
        for (int j = 0; j < x; ++j) {
            int rowIdx = i + j*y;
            for (int k = 0; k < d; ++k) {
                for(unsigned int yy =0; yy < kernelsize; ++yy)
                    for(unsigned int xx=0; xx < kernelsize; ++xx)
                    {
                        // here take care of the transpose in the original method
                        //int colIdx = xx + yy*colBlock; // this would be not transposed
                        int colIdx = xx*kernelsize + yy+d*pow(kernelsize,2);

                        col.at<float>(rowIdx,colIdx) = FeatureMaps.at(d).at<float>(i+yy, j+xx);
                    }
            }
        }
    }
}

void myaccumulate(vector<Mat> &z, vector<Mat> &conv_result) {
    int batch=conv_result.size();
    if (z.size()==0){
        for (int i = 0; i < batch; ++i) {
            z.push_back(Mat::zeros(conv_result.at(0).rows,conv_result.at(0).cols,CV_32FC1));
        }
    }

    for (int i = 0; i < batch; ++i) {
    z.at(i)=z.at(i)+conv_result.at(i);
    }
}

double calc_cross_entropy(const vector<Mat> output, const vector<Mat> &train_y) {
    double cross_entropy=0;
    int batch=output.size();
    for (int i = 0; i < batch; ++i) {
        Mat temp=output.at(i)*train_y.at(i).t();
        cross_entropy=cross_entropy-log(temp.at<float>(0,0));
    }
}

vector<Mat> derivation_fcl(const vector<Mat> &Delta_vector, const Mat &Weight) {
    int batch=Delta_vector.size();
    vector<Mat> Delta_vector_former;
    for (int i = 0; i < batch; ++i) {
        Delta_vector_former.push_back(Delta_vector.at(i)*Weight);
    }
    return Delta_vector_former;
}

vector<Mat> vector_transpose(const vector<Mat> &X_vector) {
    int batch=X_vector.size();
    vector<Mat> _Y;
    for (int i = 0; i < batch; ++i) {
        _Y.push_back(X_vector.at(i).t());
    }
    return _Y;
}

//TODO 更新权值用
vector<Mat> grad_fcl_W(const vector<Mat> &Delta_vector, const vector<Mat> &image_batch) {
    int batch=image_batch.size();
    Mat W_grad_former=Mat::zeros();
    for (int i = 0; i < batch; ++i) {
        W_grad_former=W_grad_former+image_batch.at(i)*Delta_vector_latter;
    }
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
    for (int i = 0; i < dim; i++) {
        // sigmoid function: y = exp(xi) / sum(exp(xi))
        vector_softmax.at(i) = exp(vector.at(i)) / SUM_Ex;
    }
    return vector_softmax;
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

