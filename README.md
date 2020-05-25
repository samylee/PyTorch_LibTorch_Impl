# PyTorch_LibTorch_Impl
PyTorch分类网络：Python训练_测试_模型转换 &amp;&amp; Windows_LibTorch_C++部署
# 前言
1.将PyTorch训练的网络通过模型转换，部署到Windows纯C++下执行，落地应用。  
2.这里并没有将PyTorch模型转至其他深度学习框架下，而是通过PyTorch的LibTorch来完成相关C++的部署应用。
# 相关说明
PyTorch版本：Torch-1.4.0-cu101  
LibTorch版本：LibTorch-1.4.0-cu101  
Anaconda版本：Anaconda3-Python3.6  
GPU：GTX1080  
VS版本：VS2017（用于编译LibTorch）
# 下载说明
下载LibTorch_VS2017 [百度网盘](https://pan.baidu.com/s/1TWAJuqsPqztPGbDjxnuh_A) 密码：t8is  
下载OpenCV420 [百度网盘](https://pan.baidu.com/s/132D2UB7q3WXC65kHxnSv4A) 密码：h7m1
# 训练数据
以[kaggle猫狗大战数据集](https://www.kaggle.com/c/dogs-vs-cats)为例，数据格式如下：  
1、训练数据路径：data/train/cat/*.jpg，data/train/dog/*.jpg  
2、验证数据路径：data/val/cat/*.jpg，data/val/dog/*.jpg
# 训练代码
见train.py
# 测试代码
见test.py
# 模型转换代码
见pkl2pt.py
# c++实现代码
见classification.cpp
# 相关说明
1、Pytorch默认通过PIL载入图像数据，这点很重要！  
2、需要载入ResNet预训练模型，否则训练效果较差！  
3、PyTorch训练出来的模型格式为pkl，需要将其转换为pt格式，C++方能采用torch::jit::load方式载入。  
4、转换时，有CPU和GPU两种方式，C++实现时可通过两种方式载入。  
5、C++实现时，需将PIL格式的图像转换为OpenCV的图像，否则数据不统一，导致测试结果不正确！
# 博客链接
https://blog.csdn.net/samylee
