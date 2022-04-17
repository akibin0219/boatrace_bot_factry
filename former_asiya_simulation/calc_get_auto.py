#bet_scripts直下に配置すること！！===============================================================================
#bet_scripts直下に配置すること！！===============================================================================
#bet_scripts直下に配置すること！！===============================================================================
#bet_scripts直下に配置すること！！===============================================================================
#bet_scripts直下に配置すること！！===============================================================================
#bet_scripts直下に配置すること！！===============================================================================
#bet_scripts直下に配置すること！！===============================================================================



import pandas as pd
import numpy as np
from tqdm import tqdm
import requests #クローリングのためのモジュール
from bs4 import BeautifulSoup as bs4#HTMLから特定の情報を抜き出すためのモジュール
import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import sys
import psycopg2
import time
import datetime
import os #ディレクトリ作成用
#自作のモジュールのインポート
import module.master as master
from sqlalchemy import create_engine #書き込みのエンジンをpostgreに変えるのに使う。

def undo_trans_com(target_com_arr):
    racers_arr=['1','2','3','4','5','6']
    result_com=0
    result_com+=20*racers_arr.index(target_com_arr[0])
    racers_arr.pop(racers_arr.index(target_com_arr[0]))
    result_com+=4*racers_arr.index(target_com_arr[1])
    racers_arr.pop(racers_arr.index(target_com_arr[1]))
    result_com+=racers_arr.index(target_com_arr[2])+1
    return result_com


def calc_gain(place_num,date,bet_1_money,pred_df):
    results=[]
    returns=[]
    for i in range(12):
        rno=i+1
        #まず初めに１ページの情報を抜き出す機能
        url='http://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={place_num}&hd={date}'.format(rno=rno,place_num=place_num,date=date)
        response=requests.get(url)#対象のURLをget
        response.encoding = response.apparent_encoding
        result_page=bs4(response.text, 'html.parser')
        #レース結果格納
        result_dev=result_page.select_one(".numberSet1_row")
        result_com_arr=[result_dev.select(".numberSet1_number")[0].string,result_dev.select(".numberSet1_number")[1].string,result_dev.select(".numberSet1_number")[2].string]
        result_com=undo_trans_com(result_com_arr)
        #配当金
        return_money=result_page.select_one(".is-payout1").string
        return_money=str(return_money).replace("¥",'')
        return_money=str(return_money).replace(",",'')


        results.append(result_com)
        returns.append(return_money)
        time.sleep(1)
    result_df=pd.DataFrame({'num_race':np.arange(1,13),
                  'result_coms':results,
                  'return_money':returns})

    preds=[pred_com.replace("pred_",'') for pred_com in pred_df.columns]
    total_df=pred_df.copy()
    total_df['result_coms']=result_df['result_coms']
    total_df['return_money']=result_df['return_money']
    total_df['return_money']=total_df['return_money'].astype(int)#型変換
    flags=[0]*12
    #予測の正誤判定路のフラグカラムづくり
    i=0
    for index,row in total_df.iterrows():
        for pred in preds:
            if (int(row['pred_{}'.format(pred)]==1)) and (int(pred) == row['result_coms']):
                flags[i]=1
            else:
                pass
        i+=1
    total_df['flags']=flags
    #DB書き込み用dfの作成
    log_s=pd.Series({'date': str(datetime.datetime.strptime(date, '%Y%m%d').date()),
                         'place_name':place_name,
                         'money':total_df[total_df['flags']==1]['return_money'].sum()*(bet_1_money/100),#得られた金額
                         'money_type':'get'
                        })
    log_df=pd.DataFrame(columns=log_s.index)
    log_df=log_df.append(log_s, ignore_index=True)
    #契約AWS用,db書き込み
    connection_config = {
        'user': 'watanabe',
        'password': 'Takuma406287',
        #'host': '127.0.0.1',
        'host': 'localhost',
        'port': '5432', # なくてもOK
        'database': 'boatrace_bot'
    }
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format(**connection_config))
    log_df.to_sql(name='bet_get_log_V3_1',schema='log', con=engine, if_exists='append', index=False)
    return None

#以下がコード本体=================================================================================================
#以下がコード本体=================================================================================================
#以下がコード本体=================================================================================================

#基本設定的な部分
today_datetime = datetime.date.today()
today=today_datetime.strftime('%Y%m%d')#今日の日付
bet_1_money=1000#一レースあたりの使用金額
place_master=master.get_place_master()
print(today)

for num,name in place_master.items():
    try:
        place_name=name
        place_num=num
        pred_df=pd.read_csv('/home/ubuntu/bet_bot/bot_pred/{place_name}/{date}_{place_name}_pred.csv'.format(place_name=place_name,date=today))#契約AWS用、会場ごとの本日の予測
        pred_df=pred_df.drop(["Unnamed: 0"],axis=1)#csvファイルについている名無しの列を削除
        calc_gain(place_num,today,bet_1_money,pred_df)
        print('get:',place_name)
    except FileNotFoundError:
        print('not_found_race')
