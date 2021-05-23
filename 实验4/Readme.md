# 实验4-使用python进行音频处理  
## 实验内容

1. 学习音频相关知识点，掌握MFCC特征提取步骤，使用给定的chew.wav音频文件进行特征提取。音频文件在实验群里下载。

2. 部署KALDI，简要叙述部署步骤。

   运行yes/no项目实例，简要解析发音词典内容，画出初步的WFST图（按PPT里图的形式）。

   调整并运行TIMIT项目，将命令行输出的过程与run.sh各部分进行对应，叙述顶层脚本run.sh的各部分功能（不需要解析各训练过程的详细原理）。

##  ATTENTION

具体技术细节啥的见`Content_Sexurity/src/实验4.pdf`，README.md里就不再写了。

只有任务1需要运行，直接`python get_mfcc.py `。

任务2还是得自己搭，可以照着这个网站的步骤来：[KALDI部署步骤](https://zhuanlan.zhihu.com/p/44483840)

## **task1 提取chew.wav的MFCC特征**

这是task1。

需要用到librosa库，没有的话可以`pip install librosa `，conda安装容易出问题。

只安装librosa库是不行的，还需要安装依赖包scipy、audioread、resampy、soundfile，如果没有的话运行代码会报错。因此建议考虑如下命令，缺哪个用哪个：

```
pip install audioread
pip install resampy
pip install scipy
pip install soundfile
```

详细的东西看实验报告`Content_Sexurity/src/实验4.pdf`。

代码实现在`get_mfcc.py`

## task2 部署KALDI并运行

详细的东西看实验报告`Content_Sexurity/src/实验4.pdf`。

我是在Ubuntu 18.04.5 LTS上操作的，用的软件源是Ubuntu在中国的服务器。

同时推荐几篇相关的博客，有兴趣的话（遇到什么问题的话）可以看看：

1. [Kaldi入门：yes/no项目](https://www.jianshu.com/p/09deba57f339)
2. [语音识别:模型文件介绍之FST(Kaldi)](https://zhuanlan.zhihu.com/p/74829828)
3. [Kaldi中FST的可视化-以yes/no为例](https://blog.csdn.net/u013677156/article/details/77893661)
4. [Kaldi教程（一）](http://fancyerii.github.io/kaldidoc/tutorial1/)

