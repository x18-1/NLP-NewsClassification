# 爬虫

from lxml import etree
from selenium import webdriver
import time
import csv
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")  
bro=webdriver.Chrome(executable_path="C:/Users/21/Desktop/get_dataex/chromedriver.exe")
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54',
}
#https://www.thepaper.cn/channel_25951 财经
#https://www.thepaper.cn/channel_25950 时事
#https://www.thepaper.cn/channel_122908 国际
#https://www.thepaper.cn/channel_119908  科技
# https://www.thepaper.cn/channel_25952 思想
url_list=['https://www.thepaper.cn/channel_25950','https://www.thepaper.cn/channel_122908','https://www.thepaper.cn/channel_119908','https://www.thepaper.cn/channel_25952']
url_name=['时事','国际','科技','思想']
def spider_data(url_list):
    for li in range(0,len(url_list)):
        bro.get(url_list[li])
        time.sleep(2)
        js_button = 'document.documentElement.scrollTop=100000'
        bro.maximize_window()
        for i in range(5):
            time.sleep(1)
            bro.execute_script(f'document.documentElement.scrollTop={(i+1)*1000}')
        time.sleep(2)
        page_text = bro.page_source
        tree=etree.HTML(page_text)
        li_list=tree.xpath('//*[@id="__next"]/main/div[3]/div[1]/div[2]/div[2]/div/div[1]/div')
        text_li=[]
        for i in range(0,len(li_list)):
            text=li_list[i].xpath('./div/div/div/div[1]/a/h2/text()')
            text_li.append(text)
        with open(f"C:/Users/21/Desktop/get_dataex/data/{url_name[li]}.csv",'a',encoding='utf-8',newline='') as fp:
            writer = csv.writer(fp)
            for j in text_li:
                writer.writerow(j)
#只有财经和其他网页结构不一样
def spider_data_caijin():
    bro.get('https://www.thepaper.cn/channel_25951')
    time.sleep(2)
    js_button = 'document.documentElement.scrollTop=100000'
    bro.maximize_window()
    for i in range(5):
        time.sleep(1)
        bro.execute_script(f'document.documentElement.scrollTop={(i+1)*1000}')
    time.sleep(2)
    page_text = bro.page_source
    tree=etree.HTML(page_text)
    li_list=tree.xpath('//*[@id="__next"]/main/div[3]/div[1]/div[3]/div[2]/div/div[1]/div')
    text_li=[]
    for i in range(0,len(li_list)):
        text=li_list[i].xpath('./div/div/div/div[1]/a/h2/text()')
        text_li.append(text)
    with open(f"C:/Users/21/Desktop/get_dataex/data/财经.csv",'a',encoding='utf-8',newline='') as fp:
        writer = csv.writer(fp)
        for j in text_li:
            writer.writerow(j)
spider_data_caijin()
spider_data(url_list)
bro.quit()