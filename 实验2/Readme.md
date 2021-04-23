# 实验2-文本分类实验
## 实验内容

1. 完成基于word2vec模型的文本分类任务；

2. 完成基于Naive Bayesian的文本分类任务。

要求使用python语言编写(或者自选语料库和任务，但要求必须使用word2vec和其中一种分类算法完成两次分类任务)

语料库使用群里面提供的素材或者自选。

实验报告中应写出所使用的算法基本原理。

##  ATTENTION

具体技术细节啥的见`Content_Sexurity/src/实验2.pdf`，README.md里就不再写了。

运行的话，先把data.zip解压，直接进入对应的目录下`python w2v.py`或者`python naivebayes.py`就行。可自行修改**w2v.py/naivebayes.py**里的一些参数。

结果是直接打印输出的，不会被保存到本地。

## word2vec 实现文本分类

训练好的model大小66M，超出了GitHub的上传限制（50M），所以不上传了，自己训练，很快的。

详细的东西看实验报告`Content_Sexurity/src/实验2.pdf`。

代码实现在`utils_w2v.py`中的`class TASK`，在`w2v.py`中被调用。

## 基于贝叶斯的文本分类

详细的东西看实验报告`Content_Sexurity/src/实验2.pdf`。

代码实现在`nb_utils.py`中的`class TextClassification`，在`naivebayes.py`中被调用。

用的是nltk包。