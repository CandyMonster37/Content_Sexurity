# -*- coding:utf-8 -*-
from crawl import BaiDu, Dang
import csv
import time
import random
import os


def tar_baidu(url, save=False, wait=True):
    baidu = BaiDu(url)
    if len(baidu.html) == 0:
        print('error')
        return
    for item in baidu.classes.keys():
        link = baidu.classes[item]
        title, movie, hot = baidu.proc_1_bolck(link)
        baidu.print_list(title, movie, hot)

        if wait:
            stop = random.randint(3, 7)
            time.sleep(stop)

        if save:
            with open('./data/baidu.txt', 'a', encoding='utf8') as f:
                f.write(title+'\n')
                head = baidu.tplt.format("排名", "电影名称", "搜索指数", chr(12288))
                f.write(head+'\n')
                for i in range(len(movie)):
                    line = baidu.tplt.format(i + 1, movie[i], hot[i], chr(12288))
                    f.write(line+'\n')


def tar_dang(url, save=False, wait=True):
    dang = Dang()
    for i in range(0, 25):
        page = i + 1
        link = url
        dang.update(link, page)
        dang.proc_1_page()
        if wait:
            stop = random.randint(3, 7)
            time.sleep(stop)

    dang.print_list()

    if save:
        headers = ['排名', '书名', '作者及出品方', '出版社', '出版年份', '现价', '原价', '折扣']
        rows = []
        for no in range(len(dang.book)):
            tmp = (str(no + 1), dang.book[no], dang.author[no], dang.org[no],
                   dang.year[no], dang.price[no][0], dang.price[no][1], dang.price[no][2])
            rows.append(tmp)
        with open('./data/DangDang.csv', 'a', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        with open('./data/DangDang_for_office.csv', 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)


def main():
    """
    默认对不同页面的访问有随机等待时间、默认爬取结果不保存;
    若想取消访问不同页面时的等待时间，则将wait设置为False;
    若想保存爬取到的数据，则将save设置为True;
    """
    if not os.path.exists('/data/'):
        os.mkdir('data')
    url_baidu = "http://top.baidu.com/category?c=1"
    tar_baidu(url=url_baidu, save=True, wait=True)
    url_dang = 'http://bang.dangdang.com/books/fivestars/01.28.00.00.00.00-all-0-0-2-1'
    tar_dang(url=url_dang, save=True, wait=True)


if __name__ == '__main__':
    main()
