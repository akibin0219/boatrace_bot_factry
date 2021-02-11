import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import seaborn
from pandas import DataFrame
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler #アンダーサンプリング用
import pickle
# 機械学習用
from sklearn.cluster import KMeans #クラスタリング用
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
pd.set_option('display.width',400)#勝手に改行コードを入れられるのを防ぐ


def make_PCA_df(PCA_arr):#PCAで削減したものは二次元配列で帰ってくるので、それをデータフレームにして返す関数
    X=[0]*len(PCA_arr)
    Y=[0]*len(PCA_arr)
    index=0
    for arr in PCA_arr:
        X[index]=arr[0]
        Y[index]=arr[1]
        index+=1
    return pd.DataFrame({'X':X,'Y':Y})

#バージョン1_0、()データ切り抜き関数配当金、着の情報は切りぬかなくてもうまいことやってくれる。===============================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
def data_making_1_0(df):
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
    #配当金と結果の情報を横によけておく。(仮のデータフレームに格納する）=========================================================
    #クラスタリングに使わないカラムを取り除く
    drop_col_names=['result_com','money']
    train_drops_df=pd.DataFrame(columns=drop_col_names)
    train_drops_df['result_com']=result_df['result_com']
    train_drops_df['money']=result_df['money']
    result_df=result_df.drop(drop_col_names,axis=1)
    #==============================================================================================================================
    #ダミー変数化
    #月をダミー化
    empty_arr=[0]*len(result_df)
    for number in np.arange(1,13,1):
        result_df['month_{}'.format(int(number))]=empty_arr
    dummie_df=pd.get_dummies(result_df['month'])#月をダミー化
    for column, val in dummie_df.iteritems():
        result_df['month_{}'.format(int(column))]=val


    #季節のダミー化
    empty_arr=[0]*len(result_df)
    seasons=['sp','su','au','wi']
    for season in seasons:
        result_df['{}'.format(number)]=empty_arr
    dummie_df=pd.get_dummies(result_df['season'])#季節をダミー化
    for column, val in dummie_df.iteritems():
        result_df['{}'.format(column)]=val


    #開催日数のダミー化
    empty_arr=[0]*len(result_df)
    ranges=[1,2,3,4,5,6,7]#年末がおかしくなるっぽいから、一応からむだけ作っておく　
    for number in ranges:
        result_df['range_{}'.format(int(number))]=empty_arr
    dummie_df=pd.get_dummies(result_df['range_date'])#開催日数をダミー化
    for column, val in dummie_df.iteritems():
        result_df['range_{}'.format(int(column))]=val

    #開催日数のうちの何日目かのダミー化
    empty_arr=[0]*len(result_df)
    num_dates=[1,2,3,4,5,6,7]
    for number in num_dates:
        result_df['num_date_{}'.format(int(number))]=empty_arr
    dummie_df=pd.get_dummies(result_df['num_date'])#開催日数をダミー化
    for column, val in dummie_df.iteritems():
        result_df['num_date_{}'.format(int(column))]=val

    result_df=result_df.drop(['month','season','range_date','num_date'],axis=1)#ダミー化し終わったカラムは削除する
    #レース番号のダミー化===============================================
    result_df_dummie=result_df
    race_dummie_df=pd.get_dummies(result_df_dummie['number_race'])#number_raceをダミー化
    for column, val in race_dummie_df.iteritems():
        result_df_dummie['race_{}'.format(int(column))]=val
    result_df_dummie=result_df_dummie.drop('number_race',axis=1)
    #===========================新規、性別の取り出し機能が良くなかったため作り直す
    cols=list(result_df_dummie.columns)
    male_cols=[s for s in cols if 'male' in s]#性別を示すカラムを取り出す
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

    """
    #クラスタリング
    #分けてみるクラスタの数は[8,10]の2個
    #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    target_num_cluster=[8,10]
    #test_clustaring_df=train_has_PCA_df
    clustar_target_df=result_df
    clustaring_df=clustar_target_df
    for num_cluster in target_num_cluster:
        pred = KMeans(random_state=1,n_clusters=num_cluster).fit_predict(clustar_target_df)
        clustaring_df['num={}'.format(num_cluster)]=pred
    """
    result_df['result_com']=train_drops_df['result_com']#正解ラベルを戻してあげる
    result_df['money']=train_drops_df['money']#配当金を戻してあげる
    model_df=result_df

    return model_df
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
