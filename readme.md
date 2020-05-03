###环境要求
pip install -r requirements.txt

###文件说明

#### 1.train/test
存放原始训练、测试数据

#### 2.net
存放训练后网络

#### 3.data
存放pytorch训练时dataloader所需的，寻找数据路径的txt文件

#### 4.make_dataset.py
生成pytorch训练时dataloader所需的，寻找数据路径的txt文件

#### 5.myDataLoader
用于pytorch训练的DataLoader文件
##### MyDataset类为生成单帧图片训练集
##### State类为生成单帧状态训练集
##### State_Pic类为生成单帧图片状态联合训练集
##### State_timestamp 为生成多帧状态训练集

####6 其余为网络训练文件
##### mlp.py    bp网络 单帧状态输入
##### lstm.py   lstm 单帧状态输入
##### lstm_timestamp.py  lstm 多帧状态输入
##### resnet.py  resnet网络 单帧图片输入
##### resnet_lstm.py  resnet网络 单帧图片状态联合输入

###数据说明
inp1:转向率	
inp2:车速
inp3:左边界描述1
inp4:左边界描述2
inp5:右边界描述2
inp6:右边界描述2
inp7:障碍物的相对位置x
inp8:障碍物的相对位置y
lab:方向盘转角
