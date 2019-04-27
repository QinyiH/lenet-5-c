//
// Created by assassin on 19-4-26.
//
#include "data.h"
#include <iostream>
//#include "../UnitTest_dataload/pch.h"

size_t load_image(string file_addr, vector<cv::Mat>& database)
{

    vector<cv::String> filename;
    cv::glob(file_addr, filename, false);
    size_t image_size = filename.size();
    cout << "图像集大小：" << image_size << endl;
    for (int i = 0; i < image_size; i++) {
        database.push_back(cv::imread(filename[i]));
    }
    return image_size;
}

void create_label(vector<int> image_size,vector<vector<double>>& label_set)
{
    vector<double> one_hot;
    for (int i = 0; i < 10; i++)
    {
        one_hot.assign(10, 0);
        one_hot.at(i) = 1;
        for (int j = 0; j < image_size.at(i); j++)
        {
            label_set.push_back(one_hot);
        }
    }
}

void create_database(string file_addr, vector<cv::Mat>& train_imageSet, vector<cv::Mat>& test_imageSet, vector<vector<double>>& train_labelSet, vector<vector<double>>& test_labelSet)
{
    //TODO:load_image() and create_label() in Array2D.h
    string train_set_addr = file_addr+"/trainingSet/";
    string test_set_addr = file_addr+"/testingSet/";
    stringstream s;//int转תstring
    string p = "0";
    vector<int> train_size;
    vector<int> test_size;
    int set_size = 0;
    //vector<Mat> train_data,test_data;
    for (int i = 0; i < 10; i++)
    {
        s.clear();
        s << i;
        s >> p;
        string image_addr = train_set_addr+p;
        cout << "生成train数据集:"<< p << endl;
        set_size = load_image(image_addr, train_imageSet);
        train_size.push_back(set_size);
    }
    for (int i = 0; i < 10; i++)
    {
        s.clear();
        s << i;
        s >> p;
        string image_addr = test_set_addr+p;
        cout << "生成test数据集:"<< p << endl;
        set_size = load_image(image_addr, test_imageSet);
        test_size.push_back(set_size);
    }
    create_label(train_size, train_labelSet);
    create_label(test_size,test_labelSet);
}
