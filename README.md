# 内容安全实验课
​        这个仓库用于存储内容安全实验课的代码和实验报告。

​        怎么说呢，个人觉得实验报告是最麻烦的东西。写完实验再写实验报告简直就是在写一些自己再也不想看、老师根本不会看、助教根本不细看的东西，不过是徒劳地耗费着所有人的精力和时间然后换来一个看似很重要实则没啥用的分数，说是“下义其罪，上赏其奸，上下相蒙”并不为过。至于学进去了多少、有多少人学了进去、学的东西日后又能用到多少，到底有谁心里有个准数呢，我反正是没有。

​        累了。

​        以上内容是无数个平行宇宙的同一只脑子不太够用的猴子根据**无限猴子定理**在键盘上敲出来的，**不代表任何人的真实观点，与本人无直接联系**。

##  目录

1. 实验1：爬虫实验（2021.3.30 - 2021.4.06）。
2. 实验2：文本分类实验（2021.4.13 - 2020.4.20）。
3. 实验3：使用python进行图像处理  （2021.4.28 - 2021.5.12）。
4. 实验4：使用python进行音频处理（2021.5.12-2021.5.19）。
5. 大作业：任意与内容安全有关的工作。
##  说明

代码及数据直接看对应的实验名称命名的文件夹就行。

`src`文件夹下存储的是对应实验的实验报告，PDF格式。

## 实验1-爬虫实验

​        爬取百度榜单、当当榜单。这个实验其实还算是简单，不用管反爬措施不用考虑异步加载直接爬页面然后解析就行。

## 实验2-文本分类实验

​        用word2vec和朴素贝叶斯做文本分类。直接调工具包，不用自己造轮子。

##  实验3-使用python进行图像处理  

​        2个任务，第1个用OpenCV的仿射变换实现图片缩放，第2个分别使用SVM、knn以及cnn神经网络模型实现数字图像的识别。直接调工具包，不用自己造轮子。

## 实验4-使用python进行音频处理

​        学习音频相关知识点，掌握MFCC特征提取步骤，使用给定的chew.wav音频文件进行特征提取。（注意包依赖）

​        部署KALDI，运行yes/no项目实例，调整并运行TIMIT项目。

## 大作业

​        任意与内容安全有关的工作。我做的是基于ResNet的不良图片检测（其实就是图片分类）。没有使用公开数据集，用的是自己标注的，效果有点不太好。