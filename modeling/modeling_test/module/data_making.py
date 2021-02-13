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

#バージョンとは関係のない、データの加工関数内で使う関数====================================================================================================================================================================================================
def make_PCA_df(PCA_arr):#PCAで削減したものは二次元配列で帰ってくるので、それをデータフレームにして返す関数
    X=[0]*len(PCA_arr)
    Y=[0]*len(PCA_arr)
    index=0
    for arr in PCA_arr:
        X[index]=arr[0]
        Y[index]=arr[1]
        index+=1
    return pd.DataFrame({'X':X,'Y':Y})

#dateのカラムを年だけに変換するやつ
def trans_date_type(df):
    df['date']=pd.to_datetime(df['date'])#日付が文字列なのでdateを日付型に変換
    df['year']=df['date'].dt.year
    df=df.drop('date',axis=1)
    return df

#いったん仮でいちいちほかの機能を探さないでいいようにおいておくよ（ハート）
#dateのカラムからの情報の抽出
#def trans_date_type(df):
#    df['date']=pd.to_datetime(df['date'])#日付が文字列なのでdateを日付型に変換
#    df['year']=df['date'].dt.year
#    df['month']=df['date'].dt.month
#    df['day']=df['date'].dt.day
#    df=df.drop('date',axis=1)
#    return df


#==================================================================================================================================================================
#基本的に分析用になるのかな？大会の開催日数や四半期、大会の中の何日目かの情報をもったdfを返す。
def get_event_info(df):
    df['date']=pd.to_datetime(df['date'])#日付が文字列なのでdateを日付型に変換
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['day']=df['date'].dt.day

    num_date=1
    num_date_arr=[]
    last_race_date=df['date'].values[0]#前レースの日付(処理開始時用にtarainのデータの一番初めのdateを仮に入力しておく)
    for index,row in df.iterrows():
        today_date=row['date']
        if today_date==last_race_date:#同じ日のレースだったらおなじレース日を配列に追加、次の日の日付を出力（ほぼ無操作みたいなもん）
            next_date=row['date'] + datetime.timedelta(days=1)#次の日
            num_date_arr.append(num_date)
        else:#日にちが変わった時
            if today_date==next_date:#想定していた日付（次の日のレース）だったら,レース日を一日足して、そのレース日番号を加算
                num_date+=1
                num_date_arr.append(num_date)
                last_race_date=row['date']#前回レース日を上書き
                #next_date=train_df[train_df['date']==row['date'] + datetime.timedelta(days=1)]#次の日
                next_date=row['date'] + datetime.timedelta(days=1)#次の日
                #print(next_date)
            else:#想定していた日付でない(違う大会になった)場合はレース日をリセット
                num_date=1
                num_date_arr.append(num_date)
                last_race_date=row['date']#前回レース日を上書き
                #next_date=train_df[train_df['date']==row['date'] + datetime.timedelta(days=1)]#次の日
                next_date=row['date'] + datetime.timedelta(days=1)#次の日
    df['num_date']=num_date_arr

    range_races=0#大会中の取得できたレースの数
    range_date=1#大会の開催日数
    range_date_arr=[]
    range_date_arr_2=[]#for文中で繰り返し上書きさせる用の配列
    last_race_date=df['date'].values[0]#前レースの日付(処理開始時用にtrainのデータの一番初めのdateを仮に入力しておく)
    for index,row in df.iterrows():
        today_date=row['date']
        if today_date==last_race_date:#同じ日のレースだったらおなじレース日を配列に追加、次の日の日付を出力（ほぼ無操作みたいなもん）
            range_races+=1
            next_date=row['date'] + datetime.timedelta(days=1)#次の日
            #num_date_arr.append(num_date)
        else:#日にちが変わった時
            if today_date==next_date:#想定していた日付（次の日のレース）だったら,レース日を一日足して終了
                range_date+=1
                range_races+=1
                last_race_date=row['date']#前回レース日を上書き
                #next_date=train_df[train_df['date']==row['date'] + datetime.timedelta(days=1)]#次の日
                next_date=row['date'] + datetime.timedelta(days=1)#次の日次の日
            else:#想定していた日付でない(違う大会になった)場合は現在のrange_dateをもとに前の大会のレースに大会開催日数を持たせる。

                range_date_arr_2=[range_date]*range_races
                for num in range_date_arr_2:
                    range_date_arr.append(num)
                range_races=1#大会中の取得できたレースの数
                range_date=1#大会の開催日数
                last_race_date=row['date']#前回レース日を上書き
                #next_date=train_df[train_df['date']==row['date'] + datetime.timedelta(days=1)]#次の日
                next_date=row['date'] + datetime.timedelta(days=1)#次の日
    range_date_arr_2=[range_date]*range_races#最後の日は日付の変わり絵が発生しないので特別処理
    for num in range_date_arr_2:
        range_date_arr.append(num)
    df['range_date']=range_date_arr

    #四半期カラムの作成
    df['season']=df['month']
    df['season']=df['season'].replace([3,4,5],'sp')#春
    df['season']=df['season'].replace([6,7,8],'su')#夏
    df['season']=df['season'].replace([9,10,11],'au')#秋
    df['season']=df['season'].replace([12,1,2],'wi')#冬
    #df=df.drop('date',axis=1)
    return df

#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================



#バージョン1_0、()データ切り抜き関数配当金、着の情報は切りぬかなくてもうまいことやってくれる。===============================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#train_{}のscvを突っ込むと以下の加工をする
#dateをけして、yearを追加
#各変数のダミー化
def data_making_1_0(df):

    result_df=df
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

    #クラスタリング
    #分けてみるクラスタの数は[8,10]の2個
    #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    target_num_cluster=[8,10]
    #test_clustaring_df=train_has_PCA_df
    clustar_target_df=result_df_dummie
    clustaring_df=clustar_target_df
    """
    for num_cluster in target_num_cluster:
        pred = KMeans(random_state=0,n_clusters=num_cluster).fit_predict(clustar_target_df)
        clustaring_df['num={}'.format(num_cluster)]=pred
    """
    model_df=clustaring_df
    model_df=trans_date_type(model_df)
    return model_df
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
