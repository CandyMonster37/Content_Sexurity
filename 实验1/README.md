# 实验1-爬虫实验
## 实验内容

1. 深入理解实例1-5的爬取思路及每行代码，并尝试编写。（可不写入实验报告）

2.  参考实例4，爬取百度搜索风云榜 [(Click here)](http://top.baidu.com/) 任一榜单，搜索结果按顺序逐行输出（含编号），榜单自选。  本次实验选取的目标榜单为“百度搜索风云榜-娱乐-电影榜”[(Click here)](http://top.baidu.com/category?c=1)，结果将输出并保存该页面的六个榜单：全部电影榜单、爱情榜单、喜剧榜单、惊悚榜单、科幻榜单、剧情榜单这六个板块的搜索指数排名前50的电影名称及其搜索指数。**结果将额外被保存在data目录下的txt文本文档中**。

3. 爬取当当图书排行榜（榜单自选），格式：爬取结果包含但不限于[排名 书名  作者]， 注意输出格式对齐。  

   本次实验选取的目标榜单为“当当网-图书榜-好评榜（top 500）-哲学/宗教”[(Click here)](http://bang.dangdang.com/books/fivestars/01.28.00.00.00.00-all-0-0-2-1)，结果将输出并保存宗教/哲学系列的累计好评榜排行前500本书的排名、书名、作者及出品方、出版社、出版年份、现价、原价、折扣信息。**结果将额外被保存在data目录下的csv文件中**。

##  ATTENTION

​        具体技术细节啥的见`Content_Sexurity/src/实验1.pdf`，README.md里就不再写了。运行的话直接`python main.py`就行。或者修改main函数里的url爬取其他的东西，页面布局应该不会有大的变化。

​       我没尝试爬其他榜单的东西，想试了自己改改url、爬取轮次，然后试试。应该能行。

​       要是得不到东西的话就改select或者xpath的路径。

​        data目录下有两个当当的结果csv文件，**`DangDang.csv`是`utf-8`编码，`DangDang_for_office.csv`是`utf-8-sig`编码，后者可直接用office打开查看且不会出现乱码的情况**。

## 百度搜索风云榜

简单的介绍看`BAIDU.ipynb`，用jupyter notebook打开查看。

想看详细的东西，自己看实验报告`Content_Sexurity/src/实验1.pdf`。

代码实现在`crawl.py`中的class BaiDu，在`main.py`中被调用。

## 当当图书排行榜

简单的介绍看`DANG.ipynb`，用jupyter notebook打开查看。

想看详细的东西，自己看实验报告`Content_Sexurity/src/实验1.pdf`。

代码实现在`crawl.py`中的class Dang，在`main.py`中被调用。