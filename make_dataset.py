"""
Author:Lan
用于生成处理训练过程文件，可以对文件内图片统一rename，生成 pytorch dataloader 所需的txt文件
"""
import csv
import os
import re
import numpy as np
sort = False
filePath = 'train'
fileList = os.listdir(filePath)
fileList.sort(key = lambda x:int(x))
label = []
pic_data = []
data = []
data_lstm = []
label_lstm = []
length_1 = 0
length_2 = 0
timestamp = 4

for file in fileList:
    path = os.path.join(filePath, file)
    sub_fileList = os.listdir(path)
    # print(path)
    for sub_file in sub_fileList:
        if sub_file.endswith('.csv'):
            sub_Path = os.path.join(path, sub_file)
            csv_file = csv.reader(open(sub_Path, 'r', newline=''), dialect='excel')
            line = 0
            # count = 0
            data_temp = []
            label_temp = []
            for one_line in csv_file:
                line += 1
                if line != 1:
                    if len(one_line[8]):
                        state = one_line[:-1]
                        state = [float(x) for x in state]
                        data.append(state)  # 将读取的csv分行数据按行存入列表‘date’中
                        label.append(str(float(one_line[8])/90))  # 将读取的csv分行数据按行存入列表‘date’中
                        length_1 = len(label)
                        data_temp.append(state)
                        label_temp.append(str(float(one_line[8])/90))
            for index in range(len(label_temp)-timestamp+1):
                data_lstm.append([data_temp[index], data_temp[index+1],
                                  data_temp [index+2], data_temp [index + 3],
                                  ])
                label_lstm.append(label_temp[index+timestamp-1])
        else:
            sub_Path = os.path.join(path, sub_file)
            subsub_fileList = os.listdir(sub_Path)
            if sort is True:
                # +++++++++++++++++++++++++++rename++++++++++++++++++++++++++++++++++++++++++++
                for subsub_file in subsub_fileList :
                    olddir = os.path.join(sub_Path, subsub_file)
                    filename = os.path.splitext(subsub_file) [0]
                    filetype = os.path.splitext(subsub_file) [1]
                    key = re.findall(r'[(](.*?)[)]', filename)
                    print(key)
                    if len(key) :
                        newfilename = str(key [0]) + filetype
                        newdir = os.path.join(sub_Path, newfilename)
                        os.rename(olddir, newdir)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            else:
                subsub_fileList.sort(key=lambda x: int(x[:-4]))
                for subsub_file in subsub_fileList:
                    dir = os.path.join(sub_Path, subsub_file)
                    pic_data.append(dir)
                    length_2 = len(pic_data)

    print('comparing')
    print(length_1)
    print(length_2)
    print(len(data))
    if length_1 != length_2:
        print(file)


with open('data/'+str(filePath)+'_pic.txt','w+') as f1:
    length = len(label)
    length1 = len(pic_data)
    for index in range(len(label)):
        line = pic_data[index] + ' ' + label[index] + '\n'
        f1.writelines(line)

with open('data/'+str(filePath)+'_state.txt', 'w+') as f2:
    for index in range(len(label)):
        # line = str(data[index]) + ' ' + str(label[index]) + '\n'
        line = str(['%.5f' % x for x in data[index]]) + ' ' + str(label[index]) + '\n'
        # line = str(data[index]) + ' ' + str(label[index]) + '\n'
        f2.writelines(line)

with open('data/'+str(filePath)+'_pic_state.txt','w+') as f3:
    length = len(label)
    length1 = len(data)
    for index in range(len(label)):
        line = pic_data[index] + ' ' + str(['%.10f' % x for x in data[index]]) + ' ' + label[index] + '\n'
        f3.writelines(line)

with open('data/'+str(filePath)+'_lstm_state.txt', 'w+') as f2:
    temp = []
    t = len(label_lstm)
    for index in range(len(label_lstm)):
        # line = str(data[index]) + ' ' + str(label[index]) + '\n'
        temp1 =[]
        for sub_index in range(len(data_lstm[index])):
            temp1.append(str(['%.5f' % x for x in (data_lstm [index])[sub_index]]))
        temp.append([temp1])
    for index in range(len(label_lstm)):
        line = str(temp[index]) + ' ' + str(label_lstm[index]) + '\n'
        # line = str(data[index]) + ' ' + str(label[index]) + '\n'
        f2.writelines(line)