import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import seaborn
from pandas import DataFrame
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler #アンダーサンプリング用
from sklearn.model_selection import train_test_split
import pickle
# 機械学習用
from sklearn.cluster import KMeans #クラスタリング用
from sklearn.decomposition import PCA  #次元削減用
from sklearn.ensemble import RandomForestClassifier#ランダムフォレスト
from copy import deepcopy as cp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression as LR
import time
import datetime
import os #ディレクトリ作成用
import xgboost as xgb
import sys
import psycopg2
from sqlalchemy import create_engine #書き込みのエンジンをpostgreに変えるのに使う。
import datetime as dt
import datetime
from datetime import date, timedelta

from sklearn.preprocessing import StandardScaler#モデルの評価用に標準化する関数
import scipy.stats#モデルの評価用に標準化する関数
#必要なモジュールのインポート
import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import requests #クローリングのためのモジュール
from bs4 import BeautifulSoup as bs4#HTMLから特定の情報を抜き出すためのモジュール
from webdriver_manager.chrome import ChromeDriverManager

#自作のモジュールのインポート
import module.master as master

def date_range(start, stop, step = timedelta(1)):#日付でfor文を回すためのジェネレータ
    current = start
    while current < stop:
        yield current
        current += step


def pred_th_trans_com(pred_df,th,target_com):#指定の組のカラムのみを置換。
    trans_df=pred_df.copy()
    trans_df.loc[trans_df['pred_{}'.format(target_com)] >= th, 'pred_{}'.format(target_com)] = 1
    trans_df.loc[~(trans_df['pred_{}'.format(target_com)] >=  th), 'pred_{}'.format(target_com)] = 0
    return trans_df

def get_4_section_dt(now_date):#今いる区間から直近4区間の開始日をリストで返してくれる関数
    now_sec_date=get_season_date(now_date)
    diff_sec_stdates=[0]*4
    for i in range(len(diff_sec_stdates)):
        diff_sec_stdates[3-i]=now_sec_date- relativedelta(months=3*(i+1))#古い順に日付を入れていく
    return diff_sec_stdates


def startlist_making(date,place_num):#dateと開催場所を渡してスタートリストを作成する関数
    #date='20210227'#日付を入力
    race_df=pd.DataFrame(index=[], columns=[])
    for i in range(12):
        rno=i+1
        #まず初めに１ページの情報を抜き出す機能
        url='http://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={place_num}&hd={date}'.format(rno=rno,place_num=place_num,date=date)
        response=requests.get(url)#対象のURLをget
        response.encoding = response.apparent_encoding
        start_page=bs4(response.text, 'html.parser')
        racers_div=start_page.select_one(".is-tableFixed__3rdadd")
        racers_sep_row=racers_div.find_all('tbody')
        index=0
        race_racers_data=[0]*6
        if len(racers_sep_row)==6:
            for racer_html in racers_sep_row:
                racer_row_ex_td=racer_html.find_all('td')
                #選手の登録ID
                racer_ID_div=racer_row_ex_td[2].find_all('div')
                racer_ID_div=racer_ID_div[0]
                racer_ID_div_txt=racer_ID_div.text
                start=(racer_ID_div_txt.find('/'))-35
                end=(racer_ID_div_txt.find('/')-31)
                #racer_ID=racer_ID_div_txt[start:end]#選手の登録ID
                racer_ID=racer_ID_div_txt.split('\n')[0]

                #選手のモータ番号
                racer_moter_td=racer_row_ex_td[6]
                racer_moter_td_text=racer_moter_td.text
                #racer_moter_split=racer_moter_td_text.split('\r')
                racer_moter_split=racer_moter_td_text.split('\n')
                #racer_moter_id=racer_moter_split[1][-2:]
                racer_moter_id=racer_moter_split[0]
                #選手のボート番号
                racer_boat_td=racer_row_ex_td[7]
                racer_boat_td_text=racer_boat_td.text
    #             racer_boat_split=racer_boat_td_text.split('\r')
    #             racer_boat_id=racer_boat_split[1][-2:]
                racer_boat_split=racer_boat_td_text.split('\n')
                racer_boat_id=racer_boat_split[0]

                racer_data=[racer_ID,racer_moter_id,racer_boat_id]#選手固有のデータを持ったリスト

                race_racers_data[index]=racer_data
                index+=1
        else:
            print('欠場選手等の例外が発生していないか確認してください')
        #race_racers_dataを一行に変換する
        race_row=pd.Series(index=[])#一レースの情報を一行に変換したもの
        race_num=rno#ここにはレース番号を入れる変数の値を代入
        race_row['number_race']=race_num
        for i in range(len(race_racers_data)):
            race_row['racer_{}_ID'.format(i+1)]=int(race_racers_data[i][0])
            race_row['racer_{}_mo'.format(i+1)]=int(race_racers_data[i][1])
            race_row['racer_{}_bo'.format(i+1)]=int(race_racers_data[i][2])
        race_df=race_df.append(race_row,ignore_index=True)
    return race_df


def concat_param(startlist_df,para_df):
    #クローリングしてきたスタートリストと最新の選手のパラメータを結合す。
    pred_base_df=pd.DataFrame(columns=['number_race','racer_1_ID','racer_2_ID','racer_3_ID','racer_4_ID','racer_5_ID','racer_6_ID','racer_1_rank','racer_1_male','racer_1_age','racer_1_doub','racer_1_ave_st','racer_2_rank','racer_2_male','racer_2_age','racer_2_doub','racer_2_ave_st','racer_3_rank','racer_3_male','racer_3_age','racer_3_doub','racer_3_ave_st','racer_4_rank','racer_4_male','racer_4_age','racer_4_doub','racer_4_ave_st','racer_5_rank','racer_5_male','racer_5_age','racer_5_doub','racer_5_ave_st','racer_6_rank','racer_6_male','racer_6_age','racer_6_doub','racer_6_ave_st'])
    for index,series in startlist_df.iterrows():

        #pred_base_df=pd.DataFrame(columns=['date','result_com','money','number_race','racer_1_ID','racer_2_ID','racer_3_ID','racer_4_ID','racer_5_ID','racer_6_ID','racer_1_rank','racer_1_male','racer_1_age','racer_1_doub','racer_1_ave_st','racer_2_rank','racer_2_male','racer_2_age','racer_2_doub','racer_2_ave_st','racer_3_rank','racer_3_male','racer_3_age','racer_3_doub','racer_3_ave_st','racer_4_rank','racer_4_male','racer_4_age','racer_4_doub','racer_4_ave_st','racer_5_rank','racer_5_male','racer_5_age','racer_5_doub','racer_5_ave_st','racer_6_rank','racer_6_male','racer_6_age','racer_6_doub','racer_6_ave_st'])

        #///////////////////////////////////////レースに出ているレーサーの成績を検索＆取得
        ID_1=series['racer_1_ID']
        ID_2=series['racer_2_ID']
        ID_3=series['racer_3_ID']
        ID_4=series['racer_4_ID']
        ID_5=series['racer_5_ID']
        ID_6=series['racer_6_ID']
        racer_1_df=para_df[para_df['racer_ID']==ID_1]
        racer_2_df=para_df[para_df['racer_ID']==ID_2]
        racer_3_df=para_df[para_df['racer_ID']==ID_3]
        racer_4_df=para_df[para_df['racer_ID']==ID_4]
        racer_5_df=para_df[para_df['racer_ID']==ID_5]
        racer_6_df=para_df[para_df['racer_ID']==ID_6]
        if len(racer_1_df)==1:
            pass
        else:
            racer_1_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_2_df)==1:
            pass
        else:
            racer_2_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_3_df)==1:
            pass
        else:
            racer_3_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_4_df)==1:
            pass
        else:
            racer_4_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_5_df)==1:
            pass
        else:
            racer_5_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')


        if len(racer_6_df)==1:
            pass
        else:
            racer_6_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        #追加していくデータフレームを作成

        add_df= pd.DataFrame({'number_race':series['number_race'],
                            'racer_1_ID':series['racer_1_ID'],
                            'racer_2_ID':series['racer_2_ID'],
                            'racer_3_ID':series['racer_3_ID'],
                            'racer_4_ID':series['racer_4_ID'],
                            'racer_5_ID':series['racer_5_ID'],
                            'racer_6_ID':series['racer_6_ID'],
                            'racer_1_bo':series['racer_1_bo'],
                            'racer_1_mo':series['racer_1_mo'],
                            'racer_2_bo':series['racer_2_bo'],
                            'racer_2_mo':series['racer_2_mo'],
                            'racer_3_bo':series['racer_3_bo'],
                            'racer_3_mo':series['racer_3_mo'],
                            'racer_4_bo':series['racer_4_bo'],
                            'racer_4_mo':series['racer_4_mo'],
                            'racer_5_bo':series['racer_5_bo'],
                            'racer_5_mo':series['racer_5_mo'],
                            'racer_6_bo':series['racer_6_bo'],
                            'racer_6_mo':series['racer_6_mo'],
                            'racer_1_rank':racer_1_df.iat[0,1],
                            'racer_1_male':racer_1_df.iat[0,2],
                            'racer_1_age':racer_1_df.iat[0,3],
                            'racer_1_doub':racer_1_df.iat[0,4],
                            'racer_1_ave_st':racer_1_df.iat[0,5],
                            'racer_2_rank':racer_2_df.iat[0,1],
                            'racer_2_male':racer_2_df.iat[0,2],
                            'racer_2_age':racer_2_df.iat[0,3],
                            'racer_2_doub':racer_2_df.iat[0,4],
                            'racer_2_ave_st':racer_2_df.iat[0,5],
                            'racer_3_rank':racer_3_df.iat[0,1],
                            'racer_3_male':racer_3_df.iat[0,2],
                            'racer_3_age':racer_3_df.iat[0,3],
                            'racer_3_doub':racer_3_df.iat[0,4],
                            'racer_3_ave_st':racer_3_df.iat[0,5],
                            'racer_4_rank':racer_4_df.iat[0,1],
                            'racer_4_male':racer_4_df.iat[0,2],
                            'racer_4_age':racer_4_df.iat[0,3],
                            'racer_4_doub':racer_4_df.iat[0,4],
                            'racer_4_ave_st':racer_4_df.iat[0,5],
                            'racer_5_rank':racer_5_df.iat[0,1],
                            'racer_5_male':racer_5_df.iat[0,2],
                            'racer_5_age':racer_5_df.iat[0,3],
                            'racer_5_doub':racer_5_df.iat[0,4],
                            'racer_5_ave_st':racer_5_df.iat[0,5],
                            'racer_6_rank':racer_6_df.iat[0,1],
                            'racer_6_male':racer_6_df.iat[0,2],
                            'racer_6_age':racer_6_df.iat[0,3],
                            'racer_6_doub':racer_6_df.iat[0,4],
                            'racer_6_ave_st':racer_6_df.iat[0,5] }, index=[''])
        #//////////////////////////////データフレームにadd_dfを追加していく。
        pred_base_df=pred_base_df.append(add_df)
    return pred_base_df #出走データの選手IDをもとに各種パラメータを結合したもの


def preddata_making_former_asiya(df):
    pred_race_df=df
  #pred_race_df=pred_race_df.drop(["Unnamed: 0"],axis=1)#csvファイルについている名無しの列を削除
    result_df=pred_race_df
    result_df=result_df.drop(["racer_1_ID","racer_2_ID","racer_3_ID","racer_4_ID","racer_5_ID","racer_6_ID",],axis=1)#IDはいらないので削除
    result_df=result_df.replace(0.0000,{"racer_1_ave_st_time":0.22})#新人のave_st_timeを0.22に
    result_df=result_df.replace(0.0000,{"racer_2_ave_st_time":0.22})
    result_df=result_df.replace(0.0000,{"racer_3_ave_st_time":0.22})
    result_df=result_df.replace(0.0000,{"racer_4_ave_st_time":0.22})
    result_df=result_df.replace(0.0000,{"racer_5_ave_st_time":0.22})
    result_df=result_df.replace(0.0000,{"racer_6_ave_st_time":0.22})
    result_df=result_df.replace(0.0000,{"racer_1_doub_win":0.02})#新人の着に絡む確率ave_st_timeを0.02に(新人の半期の偏差から導出)
    result_df=result_df.replace(0.0000,{"racer_2_doub_win":0.02})
    result_df=result_df.replace(0.0000,{"racer_3_doub_win":0.02})
    result_df=result_df.replace(0.0000,{"racer_4_doub_win":0.02})
    result_df=result_df.replace(0.0000,{"racer_5_doub_win":0.02})
    result_df=result_df.replace(0.0000,{"racer_6_doub_win":0.02})
  #ダミー変数化
    result_df_dummie=result_df
    race_dummie_df=pd.get_dummies(result_df_dummie['number_race'])#number_raceをダミー化
    for column, val in race_dummie_df.iteritems():
        result_df_dummie['race_{}'.format(int(column))]=val
    result_df_dummie=result_df_dummie.drop('number_race',axis=1)

    cols=list(result_df_dummie.columns)
    male_cols=[s for s in cols if 'male' in s]#性別を示すカラムを取り出す

  #===========================新規、性別の取り出し機能が良くなかったため作り直す
    empty_arr=[0]*len(result_df_dummie)
    for col in male_cols:
        for number in np.arange(0,2,1):
            result_df_dummie['{}_{}'.format(col,int(number))]=empty_arr
        male_dummie_df=pd.get_dummies(result_df_dummie[col])#性別をダミー化
        for column, val in male_dummie_df.iteritems():
            result_df_dummie['{}_{}'.format(col,int(column))]=val
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1)

    cols=list(result_df_dummie.columns)
    moter_cols=[s for s in cols if '_mo' in s]#モーター番号を示すカラムを取り出す
    boat_cols=[s for s in cols if '_bo' in s]#ボート番号を示すカラムを取り出す
    #boat もmoterも番号は1~99とする
    numbers=np.arange(1, 100, 1)
    empty_arr=[0]*len(result_df_dummie)
    for col in moter_cols:
        for number in numbers:
            result_df_dummie['{}_{}'.format(col,int(number))]=empty_arr
        moter_dummie_df=pd.get_dummies(result_df_dummie[col])#モータ番号をダミー化
        for column, val in moter_dummie_df.iteritems():
            result_df_dummie['{}_{}'.format(col,int(column))]=val
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1)

  #boat番号をダミー化
    for col in boat_cols:
        for number in numbers:
            result_df_dummie['{}_{}'.format(col,int(number))]=empty_arr
        boat_dummie_df=pd.get_dummies(result_df_dummie[col])#boat番号をダミー化
        for column, val in boat_dummie_df.iteritems():
            result_df_dummie['{}_{}'.format(col,int(column))]=val
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1)
    result_df=result_df_dummie



  #クラスタリング
  #分けてみるクラスタの数は[8,10]の2個
  #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    target_num_cluster=[8,10]
  #test_clustaring_df=train_has_PCA_df
    clustar_target_df=result_df
    clustaring_df=clustar_target_df
    for num_cluster in target_num_cluster:
        pred = KMeans(n_clusters=num_cluster).fit_predict(clustar_target_df)
        clustaring_df['num={}'.format(num_cluster)]=pred

    model_df=clustaring_df

    return model_df



def pickle_predict(pred_race_base_df):#モデルをすべてpickleで読み込んで学習データの加工、予測を行う関数
    #==============================================================================
    #学習関数で場所ごとにバージョンに対応した学習データを作る
    model_df=preddata_making_former_asiya(pred_race_base_df)

    #保存してあるモデルを読み込む(ついでに予測も行う)
    #保存してあるモデルを読み込む(ついでに予測も行う)
    #保存してあるモデルを読み込む(ついでに予測も行う)
    #保存してあるモデルを読み込む(ついでに予測も行う)
    num_race=np.arange(1,13,1)
    model3 = pickle.load(open('former_pickle_data/model_com3_dep6_per121.sav', 'rb'))
    model4 = pickle.load(open('former_pickle_data/model_com4_dep7_per131.sav', 'rb'))
    model5 = pickle.load(open('former_pickle_data/model_com5_dep8_per122.sav', 'rb'))
    model7 = pickle.load(open('former_pickle_data/model_com7_dep7_per146.sav', 'rb'))
    model13 = pickle.load(open('former_pickle_data/model_com13_dep6_per115.sav', 'rb'))
    model14= pickle.load(open('former_pickle_data/model_com14_dep4_per123.sav', 'rb'))
    model3_pred=model3.predict_proba(model_df)
    model4_pred=model4.predict_proba(model_df)
    model5_pred=model5.predict_proba(model_df)
    model7_pred=model7.predict_proba(model_df)
    model13_pred=model13.predict_proba(model_df)
    model14_pred=model14.predict_proba(model_df)
    pred_3_df=pd.DataFrame({'num_race':num_race,
                            'pred_3':[pred[1] for pred in model3_pred],
                            'pred_4':[pred[1] for pred in model4_pred],
                            'pred_5':[pred[1] for pred in model5_pred],
                            'pred_7':[pred[1] for pred in model7_pred],
                            'pred_13':[pred[1] for pred in model13_pred],
                            'pred_14':[pred[1] for pred in model14_pred]
                            })
    #print("pred_day_url",url)
    return pred_df


def trans_com(target_com):
    #racers_arr=[1,2,3,4,5,6]
    racers_arr=['1','2','3','4','5','6']
    result_com=target_com
    #result_com+=(racer_1 - 1)*20
    racer_1=int(result_com/20)+1
    #next_num=result_com%20
    next_num=result_com-(int(result_com/20)*20)
    racers_arr.remove(str(racer_1))
    racer_2=int(racers_arr[int((next_num-1)/4)])
    next_num=next_num-int(next_num/4)*4
    # if next_num==0:
    #     racer_2=racer_2-1
    racers_arr.remove(str(racer_2))

    racer_3=racers_arr[next_num-1]
    out_text='{}-{}-{}|'.format(racer_1,racer_2,racer_3)

    return out_text


def trans_pred(pred_df,place_name,version,sec_date_txt):#投票時に使えるように予測を３連単の形に変換する関数
    num_race=np.arange(1,13,1)
    buy_arr=['']*len(num_race)#レースの数
    #model_sheet_path="/home/ubuntu/bet_bot/bot_database/{place_name}/model_score_{place_name}/use_model/use_model_{place_name}_{V}.csv".format(place_name=place_name,V=version)#モデルを保存#AWS
    #model_sheet_path="bot_database/{place_name}/model_score_{place_name}/use_model/{V}/use_model_{place_name}_{date}_{V}.csv".format(place_name=place_name,V=version,date=sec_date_txt)
    model_sheet_path="/home/ubuntu/bet_bot/bot_database/{place_name}/model_score_{place_name}/use_model/{V}/use_model_{place_name}_{date}_{V}.csv".format(place_name=place_name,V=version,date=sec_date_txt)

    model_sheet_df=pd.read_csv(model_sheet_path)
    target_coms=[int(com) for com in model_sheet_df['target_com'].values]
    i=0
    for _, row in pred_df.iterrows():
        preds=''
        for_i=0
        for val in row.values:
            if val==1:
                text= trans_com(target_coms[for_i])
                preds+=text
            else:
                pass
            for_i+=1
        buy_arr[i]=preds
        i+=1
    pred_trans_df=pd.DataFrame({'num_race':num_race,
                            'buy_com':buy_arr
                            })
    return pred_trans_df

def regulation_pred_proba_scale(pred_df,target_date):
    #購買対象のレースの予測値を持ったdfと，その日の日付を渡す
    #渡したdfを昨年の予測値でーたにまぜて，予測値の差が見えやすいようにスケーリングする．関数

    #year:予測対象のレースが存在する年
    pred_df_len=len(pred_df)
    year=target_date.year#予測対象のレースの年情報だけ切り抜く
    sample_year=year-1#一年前のprobaのデータに混ぜてスケーリングを行う
    sample_proba_df=pd.read_csv('test_csv/proba_get_use_{}.csv'.format(sample_year))
    #sample_proba_df=pd.read_csv('test_csv/proba_get_use_{}.csv'.format(sample_year))
    num_race=np.arange(1,12)

    sample_proba_df=sample_proba_df.loc[:, sample_proba_df.columns.str.contains('pred')]#予測に関する列のみを抽出
    concat_target_proba_df=pd.concat([sample_proba_df, pred_df], axis=0)
    #probaをスケーリングして最大値1,最小値0にスケーリングする（pickleのモデルを使ってスケール変換機の保存を行う）
    for col in sample_proba_df.columns:
        concat_target_proba_df[col]=preprocessing.minmax_scale(concat_target_proba_df[col].values)
    #trans_target_proba_df=concat_target_proba_df[concat_target_proba_df['date']==target_date].copy()
    trans_target_proba_df=concat_target_proba_df[len(concat_target_proba_df)-pred_df_len:].copy()
    #trans_proba_df=trans_proba_df.mask(trans_proba_df>=0.5,1)
    th=0.8#中心点が0.5とは限らないっぽい，なんとなく見ながら変更をしていく
    trans_target_proba_df=trans_target_proba_df.mask(trans_target_proba_df<th,0)#データの中心は変わらないので，0.5未満は購買を行わない
    trans_target_proba_df=((trans_target_proba_df-th)*2)
    #trans_target_proba_df=(((trans_target_proba_df-th)*2)*10000)昨|日の製作時はここで係数をかけたが，実装時にはやらない.
    trans_target_proba_df=trans_target_proba_df.mask(trans_target_proba_df<=0,0)#上記の計算式だと購買を行わないものはみんな-1000となるので０に置換する
    return trans_target_proba_df

def bet_race_add_db(date,place_name,total_pred,bet_1_money):#投票を行った結果をdbに書き込む関数(レース単位でユニーク)
    place_master=master.get_place_master()
    #place_num= [k for k, v in place_master.items() if v == 'tokuyama'][0]#会場名から会場番号を出くするtips
    #log_df=pd.DataFrame({'date' : datetime.datetime.strptime(date, '%Y%m%d'),
    log_s=pd.Series({'date': str(datetime.datetime.strptime(date, '%Y%m%d').date()),
                         'place_name':place_name,
                         'money':int(total_pred*bet_1_money),
                         'money_type':'bet'
                        })
    log_df=pd.DataFrame(columns=log_s.index)
    log_df=log_df.append(log_s, ignore_index=True)
    #log_s['date']=datetime.datetime.strptime(pd.to_datetime(log_s['date']), '%Y/%m/%d %H:%M:%S').strftime('%Y/%m/%d')
    #log_s['date']=log_s['date'].split()[1]

    # connection_config = {
    #     'user': 'postgres',
    #     'password': 'admin',
    #     'host': '127.0.0.1',
    #     'port': '5432', # なくてもOK
    #     'database': 'boatrace_database'
    # }
    #ローカル用
    # connection_config = {
    #     'user': 'postgres',
    #     'password': 'Takuma406287',
    #     #'host': '127.0.0.1',
    #     'host': '127.0.0.1',
    #     'port': '5432', # なくてもOK
    #     'database': 'boatrace_bot'
    # }
    #engine = create_engine('postgresql://postgres:admin@127.0.0.1:5432/boatrace_database')#.format(**connection_config))
    engine = create_engine('postgresql://watanabe:Takuma406287@127.0.0.1:5432/boatrace_database')#raspi ubuntu server用

    #log_df.to_sql(name='former_bet_get_log_t_th05_all',schema='former', con=engine, if_exists='append', index=False)
    #log_df.to_sql(name='bet_get_log_t_V4_2_2021',schema='log', con=engine, if_exists='append', index=False)
    log_df.to_sql(name='bet_get_log_t',schema='log', con=engine, if_exists='append', index=False)
    return None

def add_BetMoney_BetFlag(regulation_df,bet_coefficient):#スケーリングしたprobaの予測値のdfと，購買金額の係数を渡して，実際の購買金額と，購買を行ったかどうかのフラグを付けてくれる関数
    bet_regulation_df=regulation_df.copy()
    bet_regulation_df=regulation_df.set_axis([col.replace('pred_','bet_') for col in regulation_df.columns],axis=1).copy()#購買金額に関連する列とわかるように名前を振りなおす
    #bet_regulation_df=((bet_regulation_df)*10000)
    bet_regulation_df=bet_regulation_df.round(-2)#投票時を想定して，桁を百円単位にま丸める（四捨五入）
    bet_regulation_df=((bet_regulation_df)*bet_coefficient)
    #bet_proba_df=bet_proba_df.mask(bet_proba_df<=0,0)#上記の計算式だと購買を行わないものはみんな-1000となるので０に置換する
    bet_flag_df=regulation_df.copy()
    #bet_flag_df=bet_flag_df.mask(bet_flag_df>=th,1).copy()#データの中心は変わらない,かつ中心以上により購買を行ったものにはフラグ付けを行う
    bet_flag_df=bet_flag_df.set_axis([col.replace('pred_','buy_flag_') for col in bet_flag_df.columns],axis=1)#購買フラグに関連する列とわかるように名前を振りなおす
    bet_flag_df=bet_flag_df.mask(bet_flag_df>0,1).copy()#データの中心は変わらない,かつ中心以上により購買を行ったものにはフラグ付けを行う
    proba_bet_flag_df=pd.concat([regulation_df,bet_regulation_df],axis=1)
    proba_bet_flag_df=pd.concat([proba_bet_flag_df,bet_flag_df],axis=1)
    #あたったレースにフラグを付ける＆獲得できた配当金の計算（レース単位でユニーク．前に出てきたものとしょりは　似ているが同じではない．）
    return proba_bet_flag_df





def bet_date_add_db(date,place_name,total_pred,bet_1_money):#投票を行った結果をdbに書き込む関数(日付単位でユニーク)
    place_master=master.get_place_master()
    #place_num= [k for k, v in place_master.items() if v == 'tokuyama'][0]#会場名から会場番号を出くするtips
    #log_df=pd.DataFrame({'date' : datetime.datetime.strptime(date, '%Y%m%d'),
    log_s=pd.Series({'date': str(datetime.datetime.strptime(date, '%Y%m%d').date()),
                         'place_name':place_name,
                         'money':int(total_pred*bet_1_money),
                         'money_type':'bet'
                        })
    log_df=pd.DataFrame(columns=log_s.index)
    log_df=log_df.append(log_s, ignore_index=True)
    #log_s['date']=datetime.datetime.strptime(pd.to_datetime(log_s['date']), '%Y/%m/%d %H:%M:%S').strftime('%Y/%m/%d')
    #log_s['date']=log_s['date'].split()[1]

    # connection_config = {
    #     'user': 'postgres',
    #     'password': 'admin',
    #     'host': '127.0.0.1',
    #     'port': '5432', # なくてもOK
    #     'database': 'boatrace_database'
    # }
    #ローカル用
    # connection_config = {
    #     'user': 'postgres',
    #     'password': 'Takuma406287',
    #     #'host': '127.0.0.1',
    #     'host': '127.0.0.1',
    #     'port': '5432', # なくてもOK
    #     'database': 'boatrace_bot'
    # }
    #engine = create_engine('postgresql://postgres:admin@127.0.0.1:5432/boatrace_database')#.format(**connection_config))
    engine = create_engine('postgresql://watanabe:Takuma406287@127.0.0.1:5432/boatrace_database')#raspi ubuntu server用

    #log_df.to_sql(name='former_bet_get_log_t_th05_all',schema='former', con=engine, if_exists='append', index=False)
    #log_df.to_sql(name='bet_get_log_t_V4_2_2021',schema='log', con=engine, if_exists='append', index=False)
    log_df.to_sql(name='bet_get_log_t',schema='log', con=engine, if_exists='append', index=False)
    return None


def split_pred(row):
    pred_coms=row['buy_com'].split("|")
    pred_coms.pop(-1)#末尾は必ず空白になるのでこれを削除
    pred_coms_split=[0]*len(pred_coms)
    for i in range(len(pred_coms)):
        pred_coms_split[i]=pred_coms[i].split("-")
    return pred_coms_split

## ブラウザを移動しながら投票を行う関数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
def auto_betting(place_num,pred_trans_df,total_pred,bet_1_money):
    # # place_num:開催会場の番号
    # # pred_trans_df:予測結果を３連単の形に直したもの
    # # total_pred:合計の予測数
    # # bet_1_money:一つの予測のbet金額
    # options = Options()
    # options.binary_location = '/usr/bin/google-chrome'
    # options.add_argument('--headless')#ヘッドレス化
    # driver = webdriver.Chrome(ChromeDriverManager().install(),chrome_options=options)
    #
    #
    # #driver = webdriver.Chrome(chrome_options=options)
    #
    # driver.get("https://ib.mbrace.or.jp/")
    #
    # # ID/PASSを入力
    # member_no = driver.find_element_by_id("memberNo")#会員番号入力フォーム
    # member_no.send_keys("08706180")#member_no.send_keys("08706180")
    # password = driver.find_element_by_id("pin")
    # password.send_keys("0219")#パスワード入力フォーム
    # password_2 = driver.find_element_by_id("authPassword")#認証番号入力フォーム
    # password_2.send_keys("tQ5S8H")#認証番号
    #
    # # ログインボタンをクリック
    # login_button = driver.find_element_by_id("loginButton")
    # login_button.click()
    #
    # time.sleep(5)
    # #最も新しく開かれた別のタブに切り替え
    # driver.switch_to.window(driver.window_handles[-1])
    #
    # #入金機能部分########################################################################################
    # #########################################################################################
    # #入金のためのタブを開く
    # charge_payoff = driver.find_element_by_id("gnavi01")
    # charge_payoff.click()
    # charge = driver.find_element_by_id("charge")
    # charge.click()
    # time.sleep(0.3)# 一回htmlが変化するために待つ
    # # 入金を行う
    # input_money = driver.find_element_by_id("chargeInstructAmt")#金額入力フォーム
    # input_money.send_keys(int((total_pred*bet_1_money)/1000))#金額
    # #input_money.send_keys("0")#金額
    #
    # input_money_pass = driver.find_element_by_id("chargeBetPassword")#認証番号入力フォーム
    # input_money_pass.send_keys("taku02")
    # charge_enter = driver.find_element_by_id("executeCharge")
    # charge_enter.click()
    # time.sleep(0.3)
    #
    # final_charge_enter = driver.find_element_by_id("ok")
    # final_charge_enter.click()
    # time.sleep(0.3)
    # close = driver.find_element_by_id("closeChargecomp")#閉じるボタン
    # close.click()
    #
    # #入金完了後の実際のbet部分
    # place_buttun=driver.find_element_by_id("jyo{}".format(place_num))#betを行う会場選び
    # place_buttun.click()
    # time.sleep(0.3)
    # for index,row in pred_trans_df.iterrows():
    #     race_num=row['num_race']#レース番号
    #     split_preds_arr=split_pred(row)
    #     if len(split_preds_arr)==0:#予測なしのレースはpass
    #         pass
    #     else:
    #         if race_num<10:#投票するレース選び
    #             race_buttun=driver.find_element_by_id("selRaceNo0{}".format(race_num))#betを行う会場選び
    #             race_buttun.click()
    #             time.sleep(0.3)
    #         else:#投票するレース選び
    #             race_buttun=driver.find_element_by_id("selRaceNo{}".format(race_num))#betを行う会場選び
    #             race_buttun.click()
    #             time.sleep(0.3)
    #         flag=0
    #         for pred in split_preds_arr:#レース内での予測の数betを行う。
    #             racer_1st_but=driver.find_element_by_id("regbtn_{}_1".format(pred[0]))#1着予測
    #             racer_1st_but.click()
    #
    #             racer_2nd_but=driver.find_element_by_id("regbtn_{}_2".format(pred[1]))#2着予測
    #             racer_2nd_but.click()
    #
    #             racer_3rd_but=driver.find_element_by_id("regbtn_{}_3".format(pred[2]))#3着予測
    #             racer_3rd_but.click()
    #
    #             # 入金を行う
    #             if flag==1:
    #                 pass
    #             else:
    #                 input_com_bet = driver.find_element_by_id("amount")#組一つあたりの金額金額入力フォーム
    #                 input_com_bet.send_keys(str(int(bet_1_money/100)))#金額(00が初めからついているので想定金額を100で割る。)
    #
    #             add_list_but=driver.find_element_by_id("regAmountBtn")#投票リストに追加
    #             add_list_but.click()
    #             time.sleep(0.3)
    #             flag=1
    #
    # input_end_but=driver.find_elements_by_class_name("btnSubmit ")[0]#投票入力完了ボタン
    # input_end_but.click()
    # time.sleep(0.3)
    #
    # total_bet = driver.find_element_by_id("amount")#合計の金額の入力フォーム
    # total_bet.send_keys(int(total_pred*bet_1_money))#合計金額は全体の組の数×一つのcomあたりの金額
    #
    # total_bet_pass = driver.find_element_by_id("pass")#合計の金額の入力フォーム
    # total_bet_pass.send_keys("taku02")#合計金額は全体の組の数×一つのcomあたりの金額
    #
    # bet_enter_but=driver.find_element_by_id("submitBet")#投票するボタン
    # bet_enter_but.click()
    # time.sleep(0.3)
    # time.sleep(10)
    # ok_but=driver.find_element_by_id("ok")#投票するボタン
    # ok_but.click()
    #
    # time.sleep(0.2)
    # close_but=driver.find_element_by_id("modifyJyoBetForm")#場をへんこうして　投票するボタン（閉じる）
    # close_but.click()
    # driver.quit()#タブを閉じる。

    return None

def bet(date,place_num,para_df,bet_1_money,verion):#これより上の自動化においてのすべての機能をまとめた関数、クロール、予測、投票までのすべてを行う。
    #try:#会場名ごとに、本日の開催があるのか銅貨を判別する
    date_txt=date.strftime('%Y%m%d')
    now_sec_date=get_season_date(date)#今いる日付が所属する区間の開始日を取得
    now_sec_date_txt=now_sec_date.strftime('%Y%m%d')

    print(date_txt)
    print(place_num)
    #try:
        #version='V3_1'
    place_master=master.get_place_master()
    startlist_df=startlist_making(date_txt,place_num)#クローリング
    pred_base_df=concat_param(startlist_df,para_df)#選手のパラメータを結合
    for i in range(6):#スクレイピングの結果だとなぜかrankのデータ型が変わってしまうので整形する
        pred_base_df["racer_{}_rank".format(i+1)]= pred_base_df["racer_{}_rank".format(i+1)].astype(int)
    place_name=place_master[place_num]
    pred_df=pickle_predict(pred_base_df,date,now_sec_date,place_name,version)
    #会場ごとの一日の合計予測数
    pred_trans_df=trans_pred(pred_df,place_name,version,now_sec_date_txt)#３連単の形に戻す
    total_pred=pred_df.sum().sum()#全体の予測数
    total_use=pred_df.sum().sum()*bet_1_money#会場ごとの一日の合計予測数からの使用金額

    bet_add_db(date,place_name,total_pred,bet_1_money)#dbにログを書き込む
    auto_betting(place_num,pred_trans_df,total_pred,bet_1_money)
    print(date,'_bet_',place_name)
    # except:
    #     print("not_found_race_today")
    return None



# #バックテスト用コード
# #para_df=pd.read_csv('bot_database/racer_para/21/21.csv')#ローカル用
# para_df=pd.read_csv('/home/ubuntu/bet_bot/bot_database/racer_para/21/21.csv')#契約AWS，ラズパイ用
#
# para_df=para_df.drop(["Unnamed: 0"],axis=1)#csvファイルについている名無しの列を削除
# version='V4_2'
# #for pred_date in date_range(date(2021, 1,1), date(2021, 10, 31)):
# for pred_date in date_range(date(2021, 1,2), date(2021, 9, 30)):
#
#     today=pred_date
#
#     #以下がコード本体=================================================================================================
#     #以下がコード本体=================================================================================================
#     #以下がコード本体=================================================================================================
#     #now_sec_date_txt=now_sec_date.strftime('%Y%m%d')#今日の日付を文字列に
#     bet_1_money=1000#一レースあたりの使用金額
#     place_master=master.get_place_master()
#
#
#     #バックテストのみでの処理
#     place_names=place_master.values
#     place_nums=list(place_master.keys())
#     #for num,name in place_master.items():
#     for num in place_nums[:10]:
#
#         #try:
#         #place_name=name
#         #place_name=place_master[num]
#         place_num=num
#         bet(today,place_num,para_df,bet_1_money,version)
#         # except:
#         #     print('error')
