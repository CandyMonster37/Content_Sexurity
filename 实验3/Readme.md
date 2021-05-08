# 实验3-使用python进行图像处理  
## 实验内容

1. 自己找一张图片，用OpenCV的仿射变换实现图片缩放。

2. 理解HOG、ORC过程，修改digits.py或独立编程，实现数字图像的识别，要求分别使用SVM、knn（第六次课）以及cnn神经网络模型（第七次课），以群文件中digit_data.rar中train文件夹下数据作为训练集，test文件夹下图片作为测试集，并计算测试集上分类正确率，并自己找测试数据（可以是一个或多个数字）。

实验报告中应写出所使用的算法基本原理。

##  ATTENTION

具体技术细节啥的见`Content_Sexurity/src/实验3.pdf`，README.md里就不再写了。

运行的话，先把data.zip解压，直接进入对应的目录下**后面的没写**

结果是直接打印输出的，不会被保存到本地。

## **task1 用OpenCV的仿射变换实现图片缩放**

这是task1。进文件夹里可以修改一些参数，比如缩放倍数、是否调整窗口大小以使其适应图片等。

需要用到openCV库，没有的话可以考虑`pip install opencv-python `

详细的东西看实验报告`Content_Sexurity/src/实验3.pdf`。

代码实现在`/task1/task1.py`，测试图片是test.jpg，可以自己修改图片然后修改代码里的文件名，然后进入task1文件夹中`python task1.py`

## task2 hog + svm数字图像的识别

详细的东西看实验报告`Content_Sexurity/src/实验3.pdf`。

做了HOG+SVM，代码实现在`/task2/task2.py`，可以训练模型和测试测试集数据，写了个人检测，效果好像不太理想。跟图片质量有很大关系，前景色和后景色差别越明显越容易识别出来。

## **task3 knn做数字识别**

详细的东西看实验报告`Content_Sexurity/src/实验3.pdf`。