//
// Created by assassin on 19-4-26.
//

#include "maths.h"

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
