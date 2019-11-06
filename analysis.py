#!/usr/bin/env python
# coding: utf-8

# # python数据分析实践教程

# **本教程抓取链家网新房房源信息，存储到本地，并分析房价，房子标签等，利用echarts可视化**

# ## 网络数据获取
# ### 第1步 导入用到的包

# In[1]:


import os
import requests
from lxml import etree
import random
import json
import pandas as pd
from pandas.io.json import json_normalize
import math
import re


# ### 第2步  创建存储数据的目录

# In[2]:


os.makedirs('./loupan',exist_ok=True)


# ### 第3步 编写爬虫函数

# #### 3.1 使用多个Agent防止被封，写一个agent函数，每个随机使用agent列表中的一个

# In[3]:



# 随机获取一个UserAgent
def getUserAgent():
    UA_list = [
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like Mac OS X) App leWebKit/537.51.2 (KHTML, like Gecko) Version/7.0 Mobile/11D257 Safari/9537.53",
        "Mozilla/5.0 (Windows; U; Windows NT 5.2) AppleWebKit/525.13 (KHTML, like Gecko) Chrome/0.2.149.27 Safari/525.13",
        "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1) ;  QIHU 360EE)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1) ; Maxthon/3.0)",
        "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (Macintosh; U; IntelMac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1Safari/534.50",
        "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:51.0) Gecko/20100101 Firefox/51.0",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"]
    return random.choice(UA_list)


# #### 3.2 编写**利用request请求给定网址url，获得页面源码**函数

# In[4]:


# 使用requests获取HTML页面
def getHTML(url):
    headers = {
        'User-Agent': getUserAgent(),
        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    }

    try:
        web_data=requests.get(url,headers=headers,timeout=20)#超时时间为20秒
        status_code=web_data.status_code
        retry_count=0
        while(str(status_code)!='200' and retry_count<5):
            print('status code: ',status_code,' retry downloading url: ',url , ' ...')
            web_data=requests.get(url,headers=headers,timeout=20)
            status_code=web_data.status_code
            retry_count+=1
        if str(status_code)=='200':
            return web_data.content.decode('utf-8')
        else:
            return "ERROR"
    except Exception as e:
        print(e)
        return "ERROR"


# #### 3.3  编写根据城市信息获得该城市房源信息的URL，然后逐个页面爬取并解析，并写入本地，存储为csv格式。

# In[5]:


# 根据城市名获得新房信息
def getDetail(city):
    url=city[1]
    print('searching city: ',city[0],'...')
    url='https:'+url+'/loupan/'
    print('url is: ',url) #拼接成完整url
    
    html=getHTML(url) #下载网页
    selector=etree.HTML(html)
    
    # 获取最大页数
    try:
        maxPage=selector.xpath('//div[@class="page-box"]/@data-total-count')
        maxPage=math.ceil(int(maxPage[0])/10)
    except Exception as e:
        maxPage=1
    print('max page is: ',maxPage)
   
    df=pd.DataFrame() # 初始化dataframe
    # 每个页面分别爬取并解析
    # 先拼接出url
    # 获得url内容，利用我们之前写的getHTML函数
    # 返回的数据市json格式，我们解析json并存入pandas的dataframe
    for page in range(1,maxPage+1):
        print('fecthing page',page,'...')
        url=url+str(page)+'/?_t=1' #构造每一页的url
        result=json.loads(getHTML(url)) #获取网页内容
        df_iter=json_normalize(result['data']['list']) #格式化为dataframe
        df=df.append(df_iter) #将每一页的数据拼接
    # 每个城市单独存储为一个csv文件
    file_path='./loupan/'+city[0]+'.csv'
    df.to_csv(file_path,index=False,encoding='utf-8')


# ### 第四步 启动爬虫，爬虫首先获得城市名字，根据城市名字调用上边编写的getDetail函数，爬取并保存该城市的房源信息。
# paste to the code cell
# ```python
# url='https://gz.fang.lianjia.com/loupan/pg1/'
# html=getHTML(url)
# selector=etree.HTML(html)
# cities_url_template=selector.xpath('//div[@class="city-change animated"]/div[@class="fc-main clear"]//ul//li/div//a/@href')
# cities_name=selector.xpath('//div[@class="city-change animated"]/div[@class="fc-main clear"]//ul//li/div//a/text()')
# cities=list(zip(cities_name,cities_url_template))
# for city in cities:
#     #city 是一个元组 (城市名，城市url)
#     getDetail(city)
# ```

# In[6]:


# 获取全国所有的已知城市，逐个城市获取房源信息
#TODO


# ---
# ### 数据分析
# #### 第一步 从本地读取所有城市的房源数据，然后放入pandas的dataframe

# In[7]:


cities_name = open('./loupan/city_names.txt').readline().strip().split(',')
df=pd.DataFrame()
for city in cities_name:
    f=open('./loupan/'+city+'.csv','r',encoding='utf8')
    try:
        df_temp=pd.read_csv(f)
    except Exception as e:
        df_temp=pd.DataFrame()
    f.close()
    df_temp['city']=city
    df=df.append(df_temp)
df.to_csv('./loupan/national.csv',encoding='utf8',index=False)


# ### 第二步 分析各个城市新房数量

# In[8]:


from pyecharts import Bar
city_count_series=df.groupby('city')['url'].count().sort_values(ascending=False)
city_count_x=city_count_series.index.tolist()
city_count_y=city_count_series.values.tolist()
city_count_bar=Bar('各城市新房数量')
city_count_bar.add('',x_axis=city_count_x,y_axis=city_count_y,is_label_show=True,is_datazoom_show=True,x_rotate=30)
city_count_bar


# In[9]:


### 第三步 分析各个城市新房均价


# In[10]:


df_price_unit=df[df.show_price!=0 ]
df_price_total=df[df.total_price_start!=0]
price_avg_series=df_price_unit.groupby('city')['show_price'].mean().sort_values(ascending=False)
total_price_series=df_price_total.groupby('city')['total_price_start'].mean().sort_values(ascending=False)
price_avg_x=price_avg_series.index
price_avg_y=price_avg_series.values
total_price_x=total_price_series.index
total_price_y=total_price_series.values
price_avg_plot=Bar('各城市新房均价')
price_avg_plot.add('单位面积价格（元/平米）',x_axis=price_avg_x,y_axis=price_avg_y,is_label_show=True)
price_avg_plot.add('总价(万元/套)',x_axis=total_price_x,y_axis=total_price_y,is_label_show=True,is_datazoom_show=True,x_rotate=30)
price_avg_plot


# ### 第四步 分析新房子卧室数量

# In[11]:


import re
bedroom=[]
for index,row in df.iterrows():
    try:
        bed_num=re.findall('\d+',row.converged_rooms)[0]
    except:
        bed_num=-1
    bedroom.append(bed_num)
df['bedroom']=bedroom
bedroom_plot_x=df.bedroom.value_counts().index
bedroom_plot_y=df.bedroom.value_counts().values
from pyecharts import Pie
bedroom_plot=Pie('新房卧室数量',width=900)
bedroom_plot.add(name='卧室数量',attr=bedroom_plot_x,value=bedroom_plot_y,center=[50,60],radius=[40,80],is_random=True,rosetype='radius',is_label_show=True)
bedroom_plot


# ### 第五步 分析房子标签词云

# In[12]:


tags=[]
for index,row in df.iterrows():
    tag=row.tags
    temp=tag.lstrip('[').rstrip(']').replace("'","").replace(' ','').split(',')
    if len(temp)>0:
        tags.extend(temp)
temp=pd.DataFrame()
temp['tag']=tags
word_series=temp.tag.value_counts()
word_name=word_series.index
word_value=word_series.values
from pyecharts import WordCloud
word_tag_plot=WordCloud('房屋标签词云')
word_tag_plot.add('',word_name,word_value,shape='circle',word_size_range=[20,70])
word_tag_plot


# ### 第六步 分析价格与面积和房屋类型的关系

# In[13]:


from pyecharts import Scatter3D

mapdict={'住宅':1,'商业类':2,'底商':3,'别墅':4,'商业':5,'写字楼':6, '酒店式公寓':7}
def mapfunc(housetype):
    return mapdict[housetype]

price=df.show_price.values.tolist()
price=[i/1000 for i in price]
area=df.max_frame_area.values.tolist()
types=list(map(mapfunc,df.house_type.values))

data=[]
for i in range(len(price)):
    data.append([area[i],types[i],price[i]])
scatter=Scatter3D('价格，面积和房屋类型的关系',width=700,height=700)
scatter.add('',data,is_visualmap=True,grid3d_opacity=0.8,xaxis3d_max=650,yaxis3d_max=7,zaxis3d_max=120,
           xaxis3d_name='面积',yaxis3d_name='房屋类型',zaxis3d_name='单位价格',)
scatter


# ### 第七步 分析多个因素之间的相关性

# In[11]:


df['house_type_num']=df.house_type.apply(mapfunc)
df_heatmap=df[['average_price','avg_price_start','avg_unit_price','city_id','district_id','house_type_num','latitude','longitude','lowest_total_price','max_frame_area','min_frame_area','total_price_start']]
from pyecharts import HeatMap
heatmap=HeatMap('各因素相关性热力图',width=600,height=600)
heatmap_corr=df_heatmap.corr()
heatmap_data=[]
length=12
for i in range(length):
    for j in range(length):
        heatmap_data.append([i,j,heatmap_corr.iloc[i,j]])
heatmap.add('相关系数',heatmap_corr.columns.tolist(),heatmap_corr.columns.tolist(),heatmap_data,is_visualmap=True,visual_range=[-1,1],visual_orient='horizontal')
heatmap


# In[ ]:




