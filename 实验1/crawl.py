# -*- coding:utf-8 -*-

import requests
from bs4 import BeautifulSoup as BS
import random
import time
import re
from lxml import etree

ua_list = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 "
    "OPR/26.0.1656.60",
    "Opera/8.0 (Windows NT 5.1; U; en)",
    "Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2 ",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 "
    "Safari/534.16",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 "
    "Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 "
    "TaoBrowser/2.0 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 "
    "LBBROWSER",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR "
    "3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR "
    "3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X "
    "MetaSr 1.0",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE "
    "2.X MetaSr 1.0)",
]


def get_html(url):
    headers = {'Connection': 'close', 'User-Agent': random.choice(ua_list)}

    try:
        res = requests.get(url, headers=headers)
        print('url:', url)
        print('get response successfully! ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # print(res.status_code)
        res.encoding = res.apparent_encoding
        html = res.text
        return html
    except requests.HTTPError as e:
        print('http error: status_code', e)
        return ""
    except Exception as e:
        print('other error:')
        print(e)
        return ""


class Dang:
    def __init__(self):
        self.book_path = 'body > div.bang_wrapper > div.bang_content > div.bang_list_box > ul > li > div.name > a'
        self.author_org_path = 'body > div.bang_wrapper > div.bang_content > div.bang_list_box ' \
                               '> ul > li > div.publisher_info'
        self.html = None
        self.bs = None
        self.book = []
        self.author = []
        self.year = []
        self.org = []
        self.price = []

    def update(self, pre_url, page):
        split_ = pre_url.split('-')
        cur_url = ''
        for item in split_[:-1]:
            cur_url = cur_url + '-' + item
        cur_url = cur_url + '-' + str(page)
        fin_url = cur_url[1:]
        self.html = get_html(fin_url)
        self.bs = BS(self.html, 'html.parser')

    def proc_1_page(self):

        # 书名
        book_pre = self.bs.select(self.book_path)
        split_ = r'.*(?=（)'  # 正先行断言去除括号内的简介
        for item in book_pre:
            pre = item.text
            done = re.compile(split_).findall(pre)
            if len(done) == 0:
                done = pre
            else:
                done = done[0]
            self.book.append(done)

        # 作者信息及出版社信息
        unit_ = self.bs.select(self.author_org_path)
        for item in unit_:
            undone = item.text.strip().split('/')
            if len(undone[0]) == 0:
                author = 'None'
            else:
                author = undone[0].strip('\u3000')
            self.author.append(author)
            if len(undone[1]) == 0:
                year = 'None'
            else:
                year = undone[1].strip().strip('\r').strip('\n')
            self.year.append(year)
            if len(undone[2]) == 0:
                org = 'None'
            else:
                org = undone[2].strip('\u3000')
            self.org.append(org)

        # 原价，现价，折扣
        ht = etree.HTML(self.html)
        price_data = ht.xpath("/html/body/div/div/div/ul/li/div/p[not(@class='ebook_line')]/span")
        no = 0
        bias = 0
        while no < len(price_data):
            tmp = []
            while bias < 3:
                tmp.append(price_data[no].text)
                bias += 1
                no += 1
            bias = 0
            self.price.append(tmp)

    def print_list(self):
        head_path = 'body > div.bang_wrapper > div.layout_location > span:nth-child(7)'
        head = self.bs.select(head_path)[0].text
        tail_path = 'body > div.bang_wrapper > div.bang_title > h1'
        tail = self.bs.select(tail_path)[0].text
        title = head + tail
        print('\n'*3)
        print(title)
        print('\n')
        for no in range(len(self.book)):
            print('-' * 50)
            print('{0}：《{1}》'.format(str(no+1), self.book[no]))
            # print('书名:\t', self.book[no])
            print('作者及出品方：\t', self.author[no])
            print('出版年份：\t', self.year[no])
            print('出版社：\t', self.org[no])
            print('现价：{0}， 原价：{1}， 折扣：{2}'.format(self.price[no][0], self.price[no][1], self.price[no][2]))
            print('-' * 50)


class BaiDu:
    def __init__(self, url):
        self.url = url
        self.tplt = "{0:^5}\t{1:{3}^10}\t{2:^5}"
        self.html = get_html(self.url)
        self.cla_path = "#main > div > div.hd > h2 > a"
        self.classes = {}
        self.soup = BS(self.html, 'html.parser')
        self.get_blocks()

    def get_blocks(self):
        classes = self.soup.select(self.cla_path)

        head = "http://top.baidu.com/"
        # tail = "&c=1"  # useless

        rule = r"buzz\?b=\d*"
        for i in classes:
            link = re.compile(rule).findall(str(i))
            self.classes[i.text] = head + link[0]

    def proc_1_bolck(self, url):
        html = get_html(url)
        soup = BS(html, 'html.parser')

        path = "#main > div.mainBody > div > div > h2"
        title = soup.select(path)[0].text

        movie = []
        m_list = soup.find_all('a', 'list-title')
        for i in m_list:
            movie.append(i.text.strip())

        hot = []
        hot_list = soup.find_all('td', 'last')
        rule = r"\d*"
        for item in hot_list:
            name = re.compile(rule).findall(str(item))
            hots = -1
            for i in name:
                if len(i) == 0:
                    continue
                else:
                    hots = i
                    break
            if hots == -1:
                hots = 'None'
            hot.append(hots)

        return title, movie, hot

    def print_list(self, title, movie, hot):
        print('\n\n\n')
        print(title)
        print('\n')
        print(self.tplt.format("排名", "电影名称", "搜索指数", chr(12288)))
        for i in range(len(movie)):
            print(self.tplt.format(i + 1, movie[i], hot[i], chr(12288)))
