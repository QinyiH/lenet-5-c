#include <iostream>
#include "cmake-build-debug/include/data.h"
#include "cmake-build-debug/include/maths.h"

using namespace std;
using namespace cv;

int main() {
    string file_addr = "/home/assassin/dataset/MNIST";
    vector<Mat> train_imageset, test_imageset;
    vector<vector<double>> train_labelSet, test_labelSet;
    create_database(file_addr, train_imageset, test_imageset, train_labelSet, test_labelSet);
    cout << "train数据集总数：" << train_labelSet.size() << endl;
    cout << "test数据集总数：" << test_labelSet.size() << endl;
    return 0;
}