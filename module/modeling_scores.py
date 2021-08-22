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
from sklearn.ensemble import RandomForestClassifier#ランダムフォレスト
from copy import deepcopy as cp
from sklearn.linear_model import LogisticRegression
import time
import datetime
import os #ディレクトリ作成用
import xgboost as xgb
import sys





#modeling_scores.py:モデリングをするパラメータの決定、選定モデル作成用のスコアシートの作成を行う関数集
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
#以下は関数内で使う関数==================================================================================================================================================================================================================
def trans_result_com(target_com,trans_base_df):#comをターゲットに合わせて0,1の二値に変換する。
    #学習データのラベル変換==========================================================
    trans_df=trans_base_df.copy()
    #result_train_df=trans_base_df.copy()
    result_arr=[0]*len(trans_df)
    i=0
    for result in trans_df['result_com']:#
        if ((result==target_com)):
            result_arr[i]=1
        else:
            result_arr[i]=0
        i+=1
    trans_df['result_com']=result_arr
    return trans_df

def pred_th_trans(pred_df,th):#引数として予測結果のdeと、変換したい閾値を渡す。

    trans_df=pred_df.copy()
    trans_df.loc[trans_df['pred_proba'] >= th, 'pred'] = 1
    trans_df.loc[~(trans_df['pred_proba']  >=  th), 'pred'] = 0
    return trans_df

def pred_th_trans_com(pred_df,th,target_com):#指定の組のカラムのみを置換。
    trans_df=pred_df.copy()
    trans_df.loc[trans_df['pred_{}'.format(target_com)] >= th, 'pred_{}'.format(target_com)] = 1
    trans_df.loc[~(trans_df['pred_{}'.format(target_com)] >=  th), 'pred_{}'.format(target_com)] = 0
    return trans_df

def calc_gain(pred_gain_df):#レース単位であたっているか同課の判別と、当たった場合に得られた配当金を計算する関数
    pred_true_df=pred_gain_df[(pred_gain_df['pred']==1)&(pred_gain_df['trans_result']==1)].copy()
    pred_true_df['hit']=1
    calc_base_df=pred_gain_df.copy()
    calc_base_df['hit']=pred_true_df['hit']
    calc_base_df['gain']=pred_true_df['money']
    calc_base_df=calc_base_df.fillna(0)
    #
    #calc_base_df:予測、変換積みの結果、実際の結果、配当金、収益をすべて表したdf,合計操作は行っていない。
    #
    return calc_base_df

def check_pred_arr(pred1_df,pred2_df):#カラムの中身が同じか比較する関数
    pred_1_vals=[pred1_df[col] for col in pred1_df.columns]
    pred_2_vals=[pred2_df[col] for col in pred2_df.columns]
    for col_name1,col1,col_name2,col2 in zip(pred1_df.columns,pred_1_vals,pred2_df.columns,pred_2_vals):
        if list(col1.values)==list(col1.values):
            print(col_name1,'  and  ',col_name2,'  is same pred \n')
    return None

def pred_th_trans(pred_df,th):#閾値を渡して、その値以上を1、未満を0に置き変える。
    #引数として予測結果のdeと、変換したい閾値を渡す。
    trans_df=pred_df.copy()
    trans_df.loc[trans_df['pred_proba'] >= th, 'pred'] = 1
    trans_df.loc[~(trans_df['pred_proba']  >=  th), 'pred'] = 0
    return trans_df

#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================
#学習のためのデータ加工関数==================================================================================================================================================================================================================

def data_making_clustar_3year_expand(df,years):#モデル関連に使用するdfの作成関数、３年分に展開して行う。(クラスタリングあり、モータ番号、艇番号なし)
    result_df=df
    result_df=result_df.drop(["racer_1_ID","racer_2_ID","racer_3_ID","racer_4_ID","racer_5_ID","racer_6_ID",],axis=1)#IDはいらないので削除
    result_df=result_df.replace(0.0000,{"racer_1_ave_st_time":0.22}).copy()#新人のave_st_timeを0.22に
    result_df=result_df.replace(0.0000,{"racer_2_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_3_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_4_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_5_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_6_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_1_doub_win":0.02}).copy()#新人の着に絡む確率ave_st_timeを0.02に(新人の半期の偏差から導出)
    result_df=result_df.replace(0.0000,{"racer_2_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_3_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_4_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_5_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_6_doub_win":0.02}).copy()

    year1=years[0]
    year2=years[1]
    year3=years[2]

    #ダミー変数化
    result_df_dummie=result_df.copy()
    race_dummie_df=pd.get_dummies(result_df_dummie['number_race'])#number_raceをダミー化
    for column, val in race_dummie_df.iteritems():
        result_df_dummie['race_{}'.format(int(column))]=val
    result_df_dummie=result_df_dummie.drop('number_race',axis=1).copy()

    cols=list(result_df_dummie.columns)
    male_cols=[s for s in cols if 'male' in s]#性別を示すカラムを取り出す

    #===========================新規、性別の取り出し機能が良くなかったため作り直す
    empty_arr=[0]*len(result_df_dummie)
    for col in male_cols:
        for number in np.arange(0,2,1):
              result_df_dummie['{}_{}'.format(col,int(number))]=empty_arr.copy()
        male_dummie_df=pd.get_dummies(result_df_dummie[col]).copy()#性別をダミー化
        for column, val in male_dummie_df.iteritems():
              result_df_dummie['{}_{}'.format(col,int(column))]=val.copy()
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()

    cols=list(result_df_dummie.columns)



    moter_cols=[s for s in cols if '_mo' in s]#モーター番号を示すカラムを取り出す
    boat_cols=[s for s in cols if '_bo' in s]#ボート番号を示すカラムを取り出す

    #boat、moterの情報は使わない、
    numbers=np.arange(1, 100, 1)
    empty_arr=[0]*len(result_df_dummie)
    for col in moter_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()
    for col in boat_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()

    #クラスタリング
    #分けてみるクラスタの数は[3,5,7,9]の4個
    #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    #学習データのdateを年に変換
    result_df_dummie['date']=pd.to_datetime(result_df_dummie['date'])#日付が文字列なのでdateを日付型に変換
    result_df_dummie['year']=result_df_dummie['date'].dt.year

    #==========================================================================
    #result_df_dummie=result_df_dummie[result_df_dummie['year']!=2020].copy()#2020のデータを完全に切り離す。
    #==========================================================================

    #クラスタリングに邪魔だから消したいけど、後々使うものはいったんよけておく
    result=result_df_dummie['result_com'].values.copy()#
    money=result_df_dummie['money'].values.copy()#
    years=result_df_dummie['year'].values.copy()#

    #安全なところに移したら削除する
    result_df_dummie=result_df_dummie.drop('result_com',axis=1)
    result_df_dummie=result_df_dummie.drop('money',axis=1)
    result_df_dummie=result_df_dummie.drop('date',axis=1)
    #クラアスタリング用の学習、予測用のデータの切り分け
    clustar_final_test_df=result_df_dummie[(result_df_dummie['year']==year3)].copy()#2020のデータを最終チェックデータ(予測のターゲット)に。
    clustar_test_df = result_df_dummie[(result_df_dummie['year']==year1) | ((result_df_dummie['year']==year2) )].copy()#2018,2019のデータを検証用データに。
    clustar_train_df =  result_df_dummie[(result_df_dummie['year']!=year1) & (result_df_dummie['year']!=year2)& (result_df_dummie['year']!=year3) ].copy()#そのほかを学習データに

    #年の情報だけ切り分けに使ったからここで消す。
    clustar_final_test_df=clustar_final_test_df.drop('year',axis=1).copy()
    clustar_test_df=clustar_test_df.drop('year',axis=1).copy()
    clustar_train_df=clustar_train_df.drop('year',axis=1).copy()

    target_num_cluster=[3,5,7,9]#分けるクラスタ数によってモデルの名前を変える
    for num_cluster in target_num_cluster:
        Km = KMeans(random_state=7,n_clusters=num_cluster).fit(clustar_train_df)#rondom_stateはラッキーセブン
        final_test_pred =Km.predict(clustar_final_test_df)#rondom_stateはラッキーセブン
        test_pred =Km.predict(clustar_test_df)#rondom_stateはラッキーセブン
        train_pred = Km.predict(clustar_train_df)#rondom_stateはラッキーセブン
        #Km=========================実査に使うときはこれのモデルを会場ごとに保存して使用。
        clustar_final_test_df['num={}'.format(num_cluster)]=final_test_pred
        clustar_test_df['num={}'.format(num_cluster)]=test_pred
        clustar_train_df['num={}'.format(num_cluster)]=train_pred

    #結合して元の形に戻す。
    clustar_df=pd.concat([clustar_train_df, clustar_test_df,clustar_final_test_df]).copy()
    clustar_df['year']=years
    clustar_df['money']=money
    clustar_df['result_com']=result
    model_df=clustar_df.copy()
    return model_df


def data_making_clustar(df,years):#モデル関連に使用するdfの作成関数(クラスタリングあり、モータ番号、艇番号なし)
    result_df=df
    result_df=result_df.drop(["racer_1_ID","racer_2_ID","racer_3_ID","racer_4_ID","racer_5_ID","racer_6_ID",],axis=1)#IDはいらないので削除
    result_df=result_df.replace(0.0000,{"racer_1_ave_st_time":0.22}).copy()#新人のave_st_timeを0.22に
    result_df=result_df.replace(0.0000,{"racer_2_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_3_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_4_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_5_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_6_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_1_doub_win":0.02}).copy()#新人の着に絡む確率ave_st_timeを0.02に(新人の半期の偏差から導出)
    result_df=result_df.replace(0.0000,{"racer_2_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_3_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_4_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_5_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_6_doub_win":0.02}).copy()

    year1=years[0]
    year2=years[1]

    #ダミー変数化
    result_df_dummie=result_df.copy()
    race_dummie_df=pd.get_dummies(result_df_dummie['number_race'])#number_raceをダミー化
    for column, val in race_dummie_df.iteritems():
        result_df_dummie['race_{}'.format(int(column))]=val
    result_df_dummie=result_df_dummie.drop('number_race',axis=1).copy()

    cols=list(result_df_dummie.columns)
    male_cols=[s for s in cols if 'male' in s]#性別を示すカラムを取り出す

    #===========================新規、性別の取り出し機能が良くなかったため作り直す
    empty_arr=[0]*len(result_df_dummie)
    for col in male_cols:
        for number in np.arange(0,2,1):
              result_df_dummie['{}_{}'.format(col,int(number))]=empty_arr.copy()
        male_dummie_df=pd.get_dummies(result_df_dummie[col]).copy()#性別をダミー化
        for column, val in male_dummie_df.iteritems():
              result_df_dummie['{}_{}'.format(col,int(column))]=val.copy()
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()

    cols=list(result_df_dummie.columns)



    moter_cols=[s for s in cols if '_mo' in s]#モーター番号を示すカラムを取り出す
    boat_cols=[s for s in cols if '_bo' in s]#ボート番号を示すカラムを取り出す

    #boat、moterの情報は使わない、
    numbers=np.arange(1, 100, 1)
    empty_arr=[0]*len(result_df_dummie)
    for col in moter_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()
    for col in boat_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()

    #クラスタリング
    #分けてみるクラスタの数は[3,5,7,9]の4個
    #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    #学習データのdateを年に変換
    result_df_dummie['date']=pd.to_datetime(result_df_dummie['date'])#日付が文字列なのでdateを日付型に変換
    result_df_dummie['year']=result_df_dummie['date'].dt.year

    #==========================================================================
    #result_df_dummie=result_df_dummie[result_df_dummie['year']!=2020].copy()#2020のデータを完全に切り離す。
    #==========================================================================

    #クラスタリングに邪魔だから消したいけど、後々使うものはいったんよけておく
    result=result_df_dummie['result_com'].values.copy()#
    money=result_df_dummie['money'].values.copy()#
    years=result_df_dummie['year'].values.copy()#

    #安全なところに移したら削除する
    result_df_dummie=result_df_dummie.drop('result_com',axis=1)
    result_df_dummie=result_df_dummie.drop('money',axis=1)
    result_df_dummie=result_df_dummie.drop('date',axis=1)
    #クラアスタリング用の学習、予測用のデータの切り分け
    #clustar_final_test_df=result_df_dummie[(result_df_dummie['year']==year3)].copy()#2020のデータを最終チェックデータ(予測のターゲット)に。
    clustar_test_df = result_df_dummie[(result_df_dummie['year']==year1) | ((result_df_dummie['year']==year2) )].copy()#2019,2010のデータを検証用データに。
    clustar_train_df =  result_df_dummie[(result_df_dummie['year']!=year1) & (result_df_dummie['year']!=year2)].copy()#そのほかを学習データに

    #年の情報だけ切り分けに使ったからここで消す。
    #clustar_final_test_df=clustar_final_test_df.drop('year',axis=1).copy()
    clustar_test_df=clustar_test_df.drop('year',axis=1).copy()
    clustar_train_df=clustar_train_df.drop('year',axis=1).copy()

    target_num_cluster=[3,5,7,9]#分けるクラスタ数によってモデルの名前を変える
    for num_cluster in target_num_cluster:
        Km = KMeans(random_state=7,n_clusters=num_cluster).fit(clustar_train_df)#rondom_stateはラッキーセブン
        #final_test_pred =Km.predict(clustar_final_test_df)#rondom_stateはラッキーセブン
        test_pred =Km.predict(clustar_test_df)#rondom_stateはラッキーセブン
        train_pred = Km.predict(clustar_train_df)#rondom_stateはラッキーセブン
        #Km=========================実査に使うときはこれのモデルを会場ごとに保存して使用。
        #clustar_final_test_df['num={}'.format(num_cluster)]=final_test_pred
        clustar_test_df['num={}'.format(num_cluster)]=test_pred
        clustar_train_df['num={}'.format(num_cluster)]=train_pred

    #結合して元の形に戻す。
    #clustar_df=pd.concat([clustar_train_df, clustar_test_df,clustar_final_test_df]).copy()
    clustar_df=pd.concat([clustar_train_df, clustar_test_df]).copy()
    clustar_df['year']=years
    clustar_df['money']=money
    clustar_df['result_com']=result
    model_df=clustar_df.copy()
    return model_df


#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================
#以下は探索を行う本命関数==================================================================================================================================================================================================================

#モデルのパラメータ探索関数(XGboost)
def model_score_XGboost_3year_expand(version,years,place_name,result_df):#パラメータ探索関数(XGboostの出力を確率のやつを使用したバージョン、閾値の探索も行う。)3年間のデータを未知データとして扱い、精度予測モデル作成用の新形式のスコアシートを出力する
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    #model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'])#スコアを格納するdf

    year1=years[0]
    year2=years[1]
    year3=years[2]
    test_year1_df= result_df[(result_df['year']==year1)].copy()#2018のデータ
    test_year2_df= result_df[(result_df['year']==year2)].copy()#2018のデータ
    test_year3_df= result_df[(result_df['year']==year3)].copy()#2018のデータ

    train_df =  result_df[(result_df['year']!=year1) & (result_df['year']!=year2)& (result_df['year']!=year3) ].copy()#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    #final_test_df=final_test_df.drop(['year'],axis=1).copy()
    #test_df=test_df.drop(['year'],axis=1).copy()
    test_year1_df=test_year1_df.drop(['year'],axis=1).copy()
    test_year2_df=test_year2_df.drop(['year'],axis=1).copy()
    test_year3_df=test_year3_df.drop(['year'],axis=1).copy()

    train_df=train_df.drop(['year'],axis=1).copy()
    #金額の情報は横によけておく
    #final_test_money=pd.Series(final_test_df['money']).copy()
    #test_money=pd.Series(test_df['money']).copy()
    test_year1_money=pd.Series(test_year1_df['money']).copy()
    test_year2_money=pd.Series(test_year2_df['money']).copy()
    test_year3_money=pd.Series(test_year3_df['money']).copy()
    train_money=pd.Series(train_df['money']).copy()

    #出現数の分布
    result_com_s=train_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため

    for result_com_number in tqdm(result_com_df['result_com'].values):
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================
        result_s=result_com_df[result_com_df['result_com']==result_com]
        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_train_df=trans_result_com(result_com,result_train_df)

        result_test_year1_df=test_year1_df.copy()
        result_test_year1_df=trans_result_com(result_com,result_test_year1_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year2_df=test_year2_df.copy()
        result_test_year2_df=trans_result_com(result_com,result_test_year2_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year3_df=test_year3_df.copy()
        result_test_year3_df=trans_result_com(result_com,result_test_year3_df)#対象のラベルを１、それ以外を０に変換する関数

        result_train_df['money']=train_money
        result_test_year1_df['money']=test_year1_money
        result_test_year2_df['money']=test_year2_money
        result_test_year3_df['money']=test_year3_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,190)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        depths_arr=[5,8]

        for depth in depths_arr:#パラメータを可変していってスコアの探索を行う
            for sum_target_per in for_arr:

                index=sum_target_per-1
                target_per=100+sum_target_per#学習データを増やす
                target_per_arr[index]=target_per
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df.copy()#ベースのデータフレームをコピー
                target_1_df=target_df[target_df['result_com']==1]
                len_1=len(target_1_df)
                target_0_df=target_df[target_df['result_com']==0]
                len_0=len(target_0_df)
                target_0_df=target_0_df.iloc[(len_0-int(len_1*(target_per/100))):len_0].copy()#1に対する目標の割合ぶん0の結果だったレースを抽出（後ろから抽出）
                target_train_df=pd.concat([target_1_df, target_0_df]).copy()
                #学習＆予測ぱーと========================================================================
                #==========================================================================================================================================
                #データの切り分け
                target_x_train=target_train_df.drop('money',axis=1).copy()
                target_x_train=target_x_train.drop('result_com',axis=1)

                target_x_year1_test=result_test_year1_df.drop('money',axis=1).copy()
                target_x_year1_test=target_x_year1_test.drop('result_com',axis=1)

                target_x_year2_test=result_test_year2_df.drop('money',axis=1).copy()
                target_x_year2_test=target_x_year2_test.drop('result_com',axis=1)

                target_x_year3_test=result_test_year3_df.drop('money',axis=1).copy()
                target_x_year3_test=target_x_year3_test.drop('result_com',axis=1)

                target_y_train=target_train_df['result_com'].copy()
                target_y_year1_test=result_test_year1_df['result_com'].copy()
                target_y_year2_test=result_test_year2_df['result_com'].copy()
                target_y_year3_test=result_test_year3_df['result_com'].copy()


                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=False)#学習データ内でさらに分割してロスをもとに修正をする。

                #XGboostのデータ型に変換する
                train = xgb.DMatrix(train_x, label=train_y)#学習用
                valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                #test = xgb.DMatrix(target_x_test)#実際に使った時の利益率の算出用
                #final_test = xgb.DMatrix(target_x_final_test)#実際に使った時の利益率の算出用
                year1 = xgb.DMatrix(target_x_year1_test)#学習時のロス修正用
                year2 = xgb.DMatrix(target_x_year2_test)#学習時のロス修正用
                year3 = xgb.DMatrix(target_x_year3_test)#学習時のロス修正用
                #xgb.config_context(verbosity=0)
                param = {'max_depth': depth, #パラメータの設定
                                 'eta': 0.3,
                                 #'objective': 'binary:hinge',
                                 'objective': 'binary:logistic',#確率で出力
                                 'eval_metric': 'logloss',
                                 'verbosity':0,
                                 'subsample':0.8,
                                 'nthread':10,
                                 'gpu_id':0,
                                 'seed':7,
                                 'tree_method':'gpu_hist'
                                }
                evallist = [(train, 'train'),(valid, 'eval')]#学習時にバリデーションを監視するデータの指定。
                num_round = 800
                bst = xgb.train(param, train,num_round,evals=evallist, early_stopping_rounds=30, verbose_eval=0 )

                # 未知データに対する予測値
                #predict_y_test=bst.predict(test)
                predict_y_year1_test=bst.predict(year1)
                predict_y_year2_test=bst.predict(year2)
                predict_y_year3_test=bst.predict(year3)
                #==========================================================================================================================================
                #[1]の正答率を見る
                #pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                #                          , 'trans_result':target_y_test})
                pred_year1_test_df=pd.DataFrame({'pred_proba':predict_y_year1_test#確率分布での出力
                                                 , 'trans_result':target_y_year1_test})
                pred_year2_test_df=pd.DataFrame({'pred_proba':predict_y_year2_test#確率分布での出力
                                                 , 'trans_result':target_y_year2_test})
                pred_year3_test_df=pd.DataFrame({'pred_proba':predict_y_year3_test#確率分布での出力
                                                 , 'trans_result':target_y_year3_test})

                th_arr=[0.85,0.9,0.92]
                for th in th_arr:
                    #trans_df=pred_th_trans(pred_test_df,th)
                    #閾値をもとに予測を変換
                    year1_trans_df=pred_th_trans(pred_year1_test_df,th)
                    year2_trans_df=pred_th_trans(pred_year2_test_df,th)
                    year3_trans_df=pred_th_trans(pred_year3_test_df,th)
                    count=0
                    #/////収益計算の項
                    year1_trans_df['money']=test_year1_money
                    year1_trans_df['true_result']=test_year1_df['result_com']
                    year2_trans_df['money']=test_year2_money
                    year2_trans_df['true_result']=test_year2_df['result_com']
                    year3_trans_df['money']=test_year3_money
                    year3_trans_df['true_result']=test_year3_df['result_com']
                    #配当金の情報も考慮する。
                    #result_gain_base_df=calc_gain(trans_df)
                    year1_result_gain_base_df=calc_gain(year1_trans_df)
                    year2_result_gain_base_df=calc_gain(year2_trans_df)
                    year3_result_gain_base_df=calc_gain(year3_trans_df)



                    #scoreのseriesに情報書き込み==================
                    #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                    #model_score_s=pd.Series(dtype='float64')
                    model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
                    model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                    model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                    model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                    model_score_s['threshold']=th

                    result_gain_df_arr=[year1_result_gain_base_df,year2_result_gain_base_df,year3_result_gain_base_df]
                    year_labels=[1,2,3]
                    #年のごとのスコア情報を横に展開していく
                    for year_df,label in zip(result_gain_df_arr,year_labels):
                        if label !=3:
                            #model_score_s['総収益']=result_gain_base_df["gain"].sum()
                            model_score_s['total_get_year{year}'.format(year=label)]=year_df["gain"].sum()
                            #model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
                            model_score_s['total_use_year{year}'.format(year=label)]=100*year_df["pred"].sum()
                            #model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
                            model_score_s['num_com_year{year}'.format(year=label)]=year_df['trans_result'].sum()
                            #model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
                            model_score_s['num_pred_year{year}'.format(year=label)]=year_df['pred'].sum()
                            #model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                            model_score_s['gain_year{year}'.format(year=label)]=(model_score_s['total_get_year{year}'.format(year=label)]/model_score_s['total_use_year{year}'.format(year=label)])*100
                            #model_score_s['的中数']=result_gain_base_df['hit'].sum()
                            model_score_s['num_hit_year{year}'.format(year=label)]=year_df['hit'].sum()
                            #model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
                            model_score_s['buy_hit_per_year{year}'.format(year=label)]=(model_score_s['num_hit_year{year}'.format(year=label)]/ model_score_s['num_pred_year{year}'.format(year=label)])*100
                        else:
                            model_score_s['gain_year{year}'.format(year=label)]=(year_df["gain"].sum()/(100*year_df["pred"].sum()))*100
                    model_score_df=model_score_df.append(model_score_s,ignore_index=True,sort=False)
    #モデルの「スコアを保存
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_3year_expand_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None




def model_score_XGboost(version,years,place_name,result_df):#パラメータ探索関数(XGboostの出力を確率のやつを使用したバージョン、閾値の探索も行う。)スコアシートの形式のみ変更、データの区切りは今まで通り
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    #model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'])#スコアを格納するdf

    year1=years[0]
    year2=years[1]
    test_year1_df= result_df[(result_df['year']==year1)].copy()#2019のデータ
    test_year2_df= result_df[(result_df['year']==year2)].copy()#2020のデータ

    train_df =  result_df[(result_df['year']!=year1) & (result_df['year']!=year2)].copy()#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    #final_test_df=final_test_df.drop(['year'],axis=1).copy()
    #test_df=test_df.drop(['year'],axis=1).copy()
    test_year1_df=test_year1_df.drop(['year'],axis=1).copy()
    test_year2_df=test_year2_df.drop(['year'],axis=1).copy()

    train_df=train_df.drop(['year'],axis=1).copy()
    #金額の情報は横によけておく
    #final_test_money=pd.Series(final_test_df['money']).copy()
    #test_money=pd.Series(test_df['money']).copy()
    test_year1_money=pd.Series(test_year1_df['money']).copy()
    test_year2_money=pd.Series(test_year2_df['money']).copy()
    train_money=pd.Series(train_df['money']).copy()

    #出現数の分布
    result_com_s=train_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため

    for result_com_number in tqdm(result_com_df['result_com'].values):
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================
        result_s=result_com_df[result_com_df['result_com']==result_com]
        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_train_df=trans_result_com(result_com,result_train_df)

        result_test_year1_df=test_year1_df.copy()
        result_test_year1_df=trans_result_com(result_com,result_test_year1_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year2_df=test_year2_df.copy()
        result_test_year2_df=trans_result_com(result_com,result_test_year2_df)#対象のラベルを１、それ以外を０に変換する関数

        result_train_df['money']=train_money
        result_test_year1_df['money']=test_year1_money
        result_test_year2_df['money']=test_year2_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,190)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        depths_arr=[5,8]

        for depth in depths_arr:#パラメータを可変していってスコアの探索を行う
            for sum_target_per in for_arr:

                index=sum_target_per-1
                target_per=100+sum_target_per#学習データを増やす
                target_per_arr[index]=target_per
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df.copy()#ベースのデータフレームをコピー
                target_1_df=target_df[target_df['result_com']==1]
                len_1=len(target_1_df)
                target_0_df=target_df[target_df['result_com']==0]
                len_0=len(target_0_df)
                target_0_df=target_0_df.iloc[(len_0-int(len_1*(target_per/100))):len_0].copy()#1に対する目標の割合ぶん0の結果だったレースを抽出（後ろから抽出）
                target_train_df=pd.concat([target_1_df, target_0_df]).copy()
                #学習＆予測ぱーと========================================================================
                #==========================================================================================================================================
                #データの切り分け
                target_x_train=target_train_df.drop('money',axis=1).copy()
                target_x_train=target_x_train.drop('result_com',axis=1)

                target_x_year1_test=result_test_year1_df.drop('money',axis=1).copy()
                target_x_year1_test=target_x_year1_test.drop('result_com',axis=1)

                target_x_year2_test=result_test_year2_df.drop('money',axis=1).copy()
                target_x_year2_test=target_x_year2_test.drop('result_com',axis=1)

                target_y_train=target_train_df['result_com'].copy()
                target_y_year1_test=result_test_year1_df['result_com'].copy()
                target_y_year2_test=result_test_year2_df['result_com'].copy()

                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=False)#学習データ内でさらに分割してロスをもとに修正をする。

                #XGboostのデータ型に変換する
                train = xgb.DMatrix(train_x, label=train_y)#学習用
                valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                #test = xgb.DMatrix(target_x_test)#実際に使った時の利益率の算出用
                #final_test = xgb.DMatrix(target_x_final_test)#実際に使った時の利益率の算出用
                year1 = xgb.DMatrix(target_x_year1_test)#学習時のロス修正用
                year2 = xgb.DMatrix(target_x_year2_test)#学習時のロス修正用
                #xgb.config_context(verbosity=0)
                param = {'max_depth': depth, #パラメータの設定
                                 'eta': 0.3,
                                 #'objective': 'binary:hinge',
                                 'objective': 'binary:logistic',#確率で出力
                                 'eval_metric': 'logloss',
                                 'verbosity':0,
                                 'subsample':0.8,
                                 'nthread':10,
                                 'gpu_id':0,
                                 'seed':7,
                                 'tree_method':'gpu_hist'
                                }
                evallist = [(train, 'train'),(valid, 'eval')]#学習時にバリデーションを監視するデータの指定。
                num_round = 800
                bst = xgb.train(param, train,num_round,evals=evallist, early_stopping_rounds=30, verbose_eval=0 )

                # 未知データに対する予測値
                #predict_y_test=bst.predict(test)
                predict_y_year1_test=bst.predict(year1)
                predict_y_year2_test=bst.predict(year2)
                #==========================================================================================================================================
                #[1]の正答率を見る
                pred_year1_test_df=pd.DataFrame({'pred_proba':predict_y_year1_test#確率分布での出力
                                                 , 'trans_result':target_y_year1_test})
                pred_year2_test_df=pd.DataFrame({'pred_proba':predict_y_year2_test#確率分布での出力
                                                 , 'trans_result':target_y_year2_test})

                th_arr=[0.85,0.9,0.92]
                for th in th_arr:
                    #trans_df=pred_th_trans(pred_test_df,th)
                    #閾値をもとに予測を変換
                    year1_trans_df=pred_th_trans(pred_year1_test_df,th)
                    year2_trans_df=pred_th_trans(pred_year2_test_df,th)
                    count=0
                    #/////収益計算の項
                    year1_trans_df['money']=test_year1_money
                    year1_trans_df['true_result']=test_year1_df['result_com']
                    year2_trans_df['money']=test_year2_money
                    year2_trans_df['true_result']=test_year2_df['result_com']
                    #配当金の情報も考慮する。
                    #result_gain_base_df=calc_gain(trans_df)
                    year1_result_gain_base_df=calc_gain(year1_trans_df)
                    year2_result_gain_base_df=calc_gain(year2_trans_df)

                    #scoreのseriesに情報書き込み==================
                    #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                    #model_score_s=pd.Series(dtype='float64')
                    model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
                    model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                    model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                    model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                    model_score_s['threshold']=th

                    result_gain_df_arr=[year1_result_gain_base_df,year2_result_gain_base_df]
                    year_labels=[1,2]
                    #年のごとのスコア情報を横に展開していく
                    for year_df,label in zip(result_gain_df_arr,year_labels):
                        #model_score_s['総収益']=result_gain_base_df["gain"].sum()
                        model_score_s['total_get_year{year}'.format(year=label)]=year_df["gain"].sum()
                        #model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
                        model_score_s['total_use_year{year}'.format(year=label)]=100*year_df["pred"].sum()
                        #model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
                        model_score_s['num_com_year{year}'.format(year=label)]=year_df['trans_result'].sum()
                        #model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
                        model_score_s['num_pred_year{year}'.format(year=label)]=year_df['pred'].sum()
                        #model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                        model_score_s['gain_year{year}'.format(year=label)]=(model_score_s['total_get_year{year}'.format(year=label)]/model_score_s['total_use_year{year}'.format(year=label)])*100
                        #model_score_s['的中数']=result_gain_base_df['hit'].sum()
                        model_score_s['num_hit_year{year}'.format(year=label)]=year_df['hit'].sum()
                        #model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
                        model_score_s['buy_hit_per_year{year}'.format(year=label)]=(model_score_s['num_hit_year{year}'.format(year=label)]/ model_score_s['num_pred_year{year}'.format(year=label)])*100
                    model_score_df=model_score_df.append(model_score_s,ignore_index=True,sort=False)
    #モデルの「スコアを保存
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None



#V3_2==================================================================================================================================================================================================================
#V3_2==================================================================================================================================================================================================================
#V3_2==================================================================================================================================================================================================================
#V3_2==================================================================================================================================================================================================================

def model_score_rondom_3year_expand(version,years,place_name,result_df):#パラメータ探索関数(XGboostの出力を確率のやつを使用したバージョン、閾値の探索も行う。)3年間のデータを未知データとして扱い、精度予測モデル作成用の新形式のスコアシートを出力する
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    #model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'])#スコアを格納するdf

    year1=years[0]
    year2=years[1]
    year3=years[2]
    test_year1_df= result_df[(result_df['year']==year1)].copy()#2018のデータ
    test_year2_df= result_df[(result_df['year']==year2)].copy()#2018のデータ
    test_year3_df= result_df[(result_df['year']==year3)].copy()#2018のデータ

    train_df =  result_df[(result_df['year']!=year1) & (result_df['year']!=year2)& (result_df['year']!=year3) ].copy()#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    #final_test_df=final_test_df.drop(['year'],axis=1).copy()
    #test_df=test_df.drop(['year'],axis=1).copy()
    test_year1_df=test_year1_df.drop(['year'],axis=1).copy()
    test_year2_df=test_year2_df.drop(['year'],axis=1).copy()
    test_year3_df=test_year3_df.drop(['year'],axis=1).copy()

    train_df=train_df.drop(['year'],axis=1).copy()
    #金額の情報は横によけておく
    #final_test_money=pd.Series(final_test_df['money']).copy()
    #test_money=pd.Series(test_df['money']).copy()
    test_year1_money=pd.Series(test_year1_df['money']).copy()
    test_year2_money=pd.Series(test_year2_df['money']).copy()
    test_year3_money=pd.Series(test_year3_df['money']).copy()
    train_money=pd.Series(train_df['money']).copy()

    #出現数の分布
    result_com_s=train_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため

    for result_com_number in tqdm(result_com_df['result_com'].values):
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================
        result_s=result_com_df[result_com_df['result_com']==result_com]
        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_train_df=trans_result_com(result_com,result_train_df)

        result_test_year1_df=test_year1_df.copy()
        result_test_year1_df=trans_result_com(result_com,result_test_year1_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year2_df=test_year2_df.copy()
        result_test_year2_df=trans_result_com(result_com,result_test_year2_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year3_df=test_year3_df.copy()
        result_test_year3_df=trans_result_com(result_com,result_test_year3_df)#対象のラベルを１、それ以外を０に変換する関数

        result_train_df['money']=train_money
        result_test_year1_df['money']=test_year1_money
        result_test_year2_df['money']=test_year2_money
        result_test_year3_df['money']=test_year3_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,190)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        depths_arr=[5,8]

        for depth in depths_arr:#パラメータを可変していってスコアの探索を行う
            for sum_target_per in for_arr:

                index=sum_target_per-1
                target_per=100+sum_target_per#学習データを増やす
                target_per_arr[index]=target_per
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df.copy()#ベースのデータフレームをコピー
                target_1_df=target_df[target_df['result_com']==1]
                len_1=len(target_1_df)
                target_0_df=target_df[target_df['result_com']==0]
                len_0=len(target_0_df)
                target_0_df=target_0_df.iloc[(len_0-int(len_1*(target_per/100))):len_0].copy()#1に対する目標の割合ぶん0の結果だったレースを抽出（後ろから抽出）
                target_train_df=pd.concat([target_1_df, target_0_df]).copy()
                #学習＆予測ぱーと========================================================================
                #==========================================================================================================================================
                #データの切り分け
                target_x_train=target_train_df.drop('money',axis=1).copy()
                target_x_train=target_x_train.drop('result_com',axis=1)

                target_x_year1_test=result_test_year1_df.drop('money',axis=1).copy()
                target_x_year1_test=target_x_year1_test.drop('result_com',axis=1)

                target_x_year2_test=result_test_year2_df.drop('money',axis=1).copy()
                target_x_year2_test=target_x_year2_test.drop('result_com',axis=1)

                target_x_year3_test=result_test_year3_df.drop('money',axis=1).copy()
                target_x_year3_test=target_x_year3_test.drop('result_com',axis=1)

                target_y_train=target_train_df['result_com'].copy()
                target_y_year1_test=result_test_year1_df['result_com'].copy()
                target_y_year2_test=result_test_year2_df['result_com'].copy()
                target_y_year3_test=result_test_year3_df['result_com'].copy()


                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=False)#学習データ内でさらに分割してロスをもとに修正をする。

                #XGboostのデータ型に変換する
                # train = xgb.DMatrix(train_x, label=train_y)#学習用
                # valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                # #test = xgb.DMatrix(target_x_test)#実際に使った時の利益率の算出用
                # #final_test = xgb.DMatrix(target_x_final_test)#実際に使った時の利益率の算出用
                # year1 = xgb.DMatrix(target_x_year1_test)#学習時のロス修正用
                # year2 = xgb.DMatrix(target_x_year2_test)#学習時のロス修正用
                # year3 = xgb.DMatrix(target_x_year3_test)#学習時のロス修正用
                #xgb.config_context(verbosity=0)
                # param = {'max_depth': depth, #パラメータの設定
                #                  'eta': 0.3,
                #                  #'objective': 'binary:hinge',
                #                  'objective': 'binary:logistic',#確率で出力
                #                  'eval_metric': 'logloss',
                #                  'verbosity':0,
                #                  'subsample':0.8,
                #                  'nthread':10,
                #                  'gpu_id':0,
                #                  'seed':7,
                #                  'tree_method':'gpu_hist'
                #                 }
                # evallist = [(train, 'train'),(valid, 'eval')]#学習時にバリデーションを監視するデータの指定。
                # num_round = 800
                #bst = xgb.train(param, train,num_round,evals=evallist, early_stopping_rounds=30, verbose_eval=0 )
                clf=RandomForestClassifier(random_state=7,n_estimators=1000,max_depth=depth,n_jobs=10)
                clf=clf.fit(target_x_train,target_y_train)

                # 未知データに対する予測値
                #predict_y_year1_test=bst.predict(year1)
                #predict_y_year1_test=[arr[1] for arr in clf.predict_proba(target_x_year1_test)]
                predict_y_year1_test=clf.predict(target_x_year1_test)
                #predict_y_year2_test=bst.predict(year2)
                #predict_y_year2_test=[arr[1] for arr in clf.predict_proba(target_x_year2_test)]
                predict_y_year2_test=clf.predict(target_x_year2_test)
                #predict_y_year3_test=bst.predict(year3)
                #predict_y_year3_test=[arr[1] for arr in clf.predict_proba(target_x_year3_test)]
                predict_y_year3_test=clf.predict(target_x_year3_test)
                #==========================================================================================================================================
                #[1]の正答率を見る
                # pred_year1_test_df=pd.DataFrame({'pred_proba':predict_y_year1_test#確率分布での出力
                #                                  , 'trans_result':target_y_year1_test})
                # pred_year2_test_df=pd.DataFrame({'pred_proba':predict_y_year2_test#確率分布での出力
                #                                  , 'trans_result':target_y_year2_test})
                # pred_year3_test_df=pd.DataFrame({'pred_proba':predict_y_year3_test#確率分布での出力
                #                                  , 'trans_result':target_y_year3_test})
                pred_year1_test_df=pd.DataFrame({'pred':predict_y_year1_test#確率分布での出力
                                                 , 'trans_result':target_y_year1_test})
                pred_year2_test_df=pd.DataFrame({'pred':predict_y_year2_test#確率分布での出力
                                                 , 'trans_result':target_y_year2_test})
                pred_year3_test_df=pd.DataFrame({'pred':predict_y_year3_test#確率分布での出力
                                                 , 'trans_result':target_y_year3_test})

                #th_arr=[0.85,0.9,0.92]
                #for th in th_arr:
                #trans_df=pred_th_trans(pred_test_df,th)
                #閾値をもとに予測を変換
                # year1_trans_df=pred_th_trans(pred_year1_test_df,th)
                # year2_trans_df=pred_th_trans(pred_year2_test_df,th)
                # year3_trans_df=pred_th_trans(pred_year3_test_df,th)
                year1_trans_df=pred_year1_test_df.copy()#閾値での変換は行わない
                year2_trans_df=pred_year2_test_df.copy()#閾値での変換は行わない
                year3_trans_df=pred_year3_test_df.copy()#閾値での変換は行わない
                #/////収益計算の項
                year1_trans_df['money']=test_year1_money
                year1_trans_df['true_result']=test_year1_df['result_com']
                year2_trans_df['money']=test_year2_money
                year2_trans_df['true_result']=test_year2_df['result_com']
                year3_trans_df['money']=test_year3_money
                year3_trans_df['true_result']=test_year3_df['result_com']
                #配当金の情報も考慮する。
                #result_gain_base_df=calc_gain(trans_df)
                year1_result_gain_base_df=calc_gain(year1_trans_df)
                year2_result_gain_base_df=calc_gain(year2_trans_df)
                year3_result_gain_base_df=calc_gain(year3_trans_df)

                #scoreのseriesに情報書き込み==================
                #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                #model_score_s=pd.Series(dtype='float64')
                #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
                model_score_s=pd.Series(index=['target_com','depth','target_per','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')

                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #model_score_s['threshold']=th

                result_gain_df_arr=[year1_result_gain_base_df,year2_result_gain_base_df,year3_result_gain_base_df]
                year_labels=[1,2,3]
                #年のごとのスコア情報を横に展開していく
                for year_df,label in zip(result_gain_df_arr,year_labels):
                    if label !=3:
                        #model_score_s['総収益']=result_gain_base_df["gain"].sum()
                        model_score_s['total_get_year{year}'.format(year=label)]=year_df["gain"].sum()
                        #model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
                        model_score_s['total_use_year{year}'.format(year=label)]=100*year_df["pred"].sum()
                        #model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
                        model_score_s['num_com_year{year}'.format(year=label)]=year_df['trans_result'].sum()
                        #model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
                        model_score_s['num_pred_year{year}'.format(year=label)]=year_df['pred'].sum()
                        #model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                        model_score_s['gain_year{year}'.format(year=label)]=(model_score_s['total_get_year{year}'.format(year=label)]/model_score_s['total_use_year{year}'.format(year=label)])*100
                        #model_score_s['的中数']=result_gain_base_df['hit'].sum()
                        model_score_s['num_hit_year{year}'.format(year=label)]=year_df['hit'].sum()
                        #model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
                        model_score_s['buy_hit_per_year{year}'.format(year=label)]=(model_score_s['num_hit_year{year}'.format(year=label)]/ model_score_s['num_pred_year{year}'.format(year=label)])*100
                    else:
                        model_score_s['gain_year{year}'.format(year=label)]=(year_df["gain"].sum()/(100*year_df["pred"].sum()))*100
                model_score_df=model_score_df.append(model_score_s,ignore_index=True,sort=False)
    #モデルの「スコアを保存
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_3year_expand_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None


def model_score_rondom(version,years,place_name,result_df):#パラメータ探索関数(XGboostの出力を確率のやつを使用したバージョン、閾値の探索も行う。)スコアシートの形式のみ変更、データの区切りは今まで通り
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    #model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'])#スコアを格納するdf

    year1=years[0]
    year2=years[1]
    test_year1_df= result_df[(result_df['year']==year1)].copy()#2019のデータ
    test_year2_df= result_df[(result_df['year']==year2)].copy()#2020のデータ

    train_df =  result_df[(result_df['year']!=year1) & (result_df['year']!=year2)].copy()#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    #final_test_df=final_test_df.drop(['year'],axis=1).copy()
    #test_df=test_df.drop(['year'],axis=1).copy()
    test_year1_df=test_year1_df.drop(['year'],axis=1).copy()
    test_year2_df=test_year2_df.drop(['year'],axis=1).copy()

    train_df=train_df.drop(['year'],axis=1).copy()
    #金額の情報は横によけておく
    #final_test_money=pd.Series(final_test_df['money']).copy()
    #test_money=pd.Series(test_df['money']).copy()
    test_year1_money=pd.Series(test_year1_df['money']).copy()
    test_year2_money=pd.Series(test_year2_df['money']).copy()
    train_money=pd.Series(train_df['money']).copy()

    #出現数の分布
    result_com_s=train_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため

    for result_com_number in tqdm(result_com_df['result_com'].values):
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================
        result_s=result_com_df[result_com_df['result_com']==result_com]
        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_train_df=trans_result_com(result_com,result_train_df)

        result_test_year1_df=test_year1_df.copy()
        result_test_year1_df=trans_result_com(result_com,result_test_year1_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year2_df=test_year2_df.copy()
        result_test_year2_df=trans_result_com(result_com,result_test_year2_df)#対象のラベルを１、それ以外を０に変換する関数

        result_train_df['money']=train_money
        result_test_year1_df['money']=test_year1_money
        result_test_year2_df['money']=test_year2_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,190)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        depths_arr=[5,8]

        for depth in depths_arr:#パラメータを可変していってスコアの探索を行う
            for sum_target_per in for_arr:

                index=sum_target_per-1
                target_per=100+sum_target_per#学習データを増やす
                target_per_arr[index]=target_per
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df.copy()#ベースのデータフレームをコピー
                target_1_df=target_df[target_df['result_com']==1]
                len_1=len(target_1_df)
                target_0_df=target_df[target_df['result_com']==0]
                len_0=len(target_0_df)
                target_0_df=target_0_df.iloc[(len_0-int(len_1*(target_per/100))):len_0].copy()#1に対する目標の割合ぶん0の結果だったレースを抽出（後ろから抽出）
                target_train_df=pd.concat([target_1_df, target_0_df]).copy()
                #学習＆予測ぱーと========================================================================
                #==========================================================================================================================================
                #データの切り分け
                target_x_train=target_train_df.drop('money',axis=1).copy()
                target_x_train=target_x_train.drop('result_com',axis=1)

                target_x_year1_test=result_test_year1_df.drop('money',axis=1).copy()
                target_x_year1_test=target_x_year1_test.drop('result_com',axis=1)

                target_x_year2_test=result_test_year2_df.drop('money',axis=1).copy()
                target_x_year2_test=target_x_year2_test.drop('result_com',axis=1)

                target_y_train=target_train_df['result_com'].copy()
                target_y_year1_test=result_test_year1_df['result_com'].copy()
                target_y_year2_test=result_test_year2_df['result_com'].copy()

                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=False)#学習データ内でさらに分割してロスをもとに修正をする。
                #XGboostのデータ型に変換する
                # train = xgb.DMatrix(train_x, label=train_y)#学習用
                # valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                # #test = xgb.DMatrix(target_x_test)#実際に使った時の利益率の算出用
                # #final_test = xgb.DMatrix(target_x_final_test)#実際に使った時の利益率の算出用
                # year1 = xgb.DMatrix(target_x_year1_test)#学習時のロス修正用
                # year2 = xgb.DMatrix(target_x_year2_test)#学習時のロス修正用
                # #xgb.config_context(verbosity=0)
                # param = {'max_depth': depth, #パラメータの設定
                #                  'eta': 0.3,
                #                  #'objective': 'binary:hinge',
                #                  'objective': 'binary:logistic',#確率で出力
                #                  'eval_metric': 'logloss',
                #                  'verbosity':0,
                #                  'subsample':0.8,
                #                  'nthread':10,
                #                  'gpu_id':0,
                #                  'seed':7,
                #                  'tree_method':'gpu_hist'
                #                 }
                # evallist = [(train, 'train'),(valid, 'eval')]#学習時にバリデーションを監視するデータの指定。
                # num_round = 800
                # bst = xgb.train(param, train,num_round,evals=evallist, early_stopping_rounds=30, verbose_eval=0 )
                clf=RandomForestClassifier(random_state=7,n_estimators=1000,max_depth=depth,n_jobs=10)
                clf=clf.fit(target_x_train,target_y_train)

                # 未知データに対する予測値
                #predict_y_year1_test=bst.predict(year1)
                #predict_y_year1_test=[arr[1] for arr in clf.predict_proba(target_x_year1_test)]
                predict_y_year1_test=clf.predict(target_x_year1_test)
                #predict_y_year2_test=bst.predict(year2)
                #predict_y_year2_test=[arr[1] for arr in clf.predict_proba(target_x_year2_test)]
                predict_y_year2_test=clf.predict(target_x_year2_test)
                #==========================================================================================================================================
                #[1]の正答率を見る
                pred_year1_test_df=pd.DataFrame({'pred':predict_y_year1_test#確率分布での出力
                                                 , 'trans_result':target_y_year1_test})
                pred_year2_test_df=pd.DataFrame({'pred':predict_y_year2_test#確率分布での出力
                                                 , 'trans_result':target_y_year2_test})

                # th_arr=[0.85,0.9,0.92]
                # for th in th_arr:
                #trans_df=pred_th_trans(pred_test_df,th)
                #閾値をもとに予測を変換
                # year1_trans_df=pred_th_trans(pred_year1_test_df,th)
                # year2_trans_df=pred_th_trans(pred_year2_test_df,th)
                year1_trans_df=pred_year1_test_df.copy()#閾値での変換は行わない
                year2_trans_df=pred_year2_test_df.copy()#閾値での変換は行わない
                #/////収益計算の項
                year1_trans_df['money']=test_year1_money
                year1_trans_df['true_result']=test_year1_df['result_com']
                year2_trans_df['money']=test_year2_money
                year2_trans_df['true_result']=test_year2_df['result_com']
                #配当金の情報も考慮する。
                #result_gain_base_df=calc_gain(trans_df)
                year1_result_gain_base_df=calc_gain(year1_trans_df)
                year2_result_gain_base_df=calc_gain(year2_trans_df)

                #scoreのseriesに情報書き込み==================
                #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                #model_score_s=pd.Series(dtype='float64')
                #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
                model_score_s=pd.Series(index=['target_com','depth','target_per','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #model_score_s['threshold']=th

                result_gain_df_arr=[year1_result_gain_base_df,year2_result_gain_base_df]
                year_labels=[1,2]
                #年のごとのスコア情報を横に展開していく
                for year_df,label in zip(result_gain_df_arr,year_labels):
                    #model_score_s['総収益']=result_gain_base_df["gain"].sum()
                    model_score_s['total_get_year{year}'.format(year=label)]=year_df["gain"].sum()
                    #model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
                    model_score_s['total_use_year{year}'.format(year=label)]=100*year_df["pred"].sum()
                    #model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
                    model_score_s['num_com_year{year}'.format(year=label)]=year_df['trans_result'].sum()
                    #model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
                    model_score_s['num_pred_year{year}'.format(year=label)]=year_df['pred'].sum()
                    #model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                    model_score_s['gain_year{year}'.format(year=label)]=(model_score_s['total_get_year{year}'.format(year=label)]/model_score_s['total_use_year{year}'.format(year=label)])*100
                    #model_score_s['的中数']=result_gain_base_df['hit'].sum()
                    model_score_s['num_hit_year{year}'.format(year=label)]=year_df['hit'].sum()
                    #model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
                    model_score_s['buy_hit_per_year{year}'.format(year=label)]=(model_score_s['num_hit_year{year}'.format(year=label)]/ model_score_s['num_pred_year{year}'.format(year=label)])*100
                model_score_df=model_score_df.append(model_score_s,ignore_index=True,sort=False)
    #モデルの「スコアを保存
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None



#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================
#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================
#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================
#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================
#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================
#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================
#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================
#３年の拡張時にデータの削除を年単位でしたいときに使用する関数=============================================================================================================================================================================================




def data_making_clustar_3year_expand_select_year(df,years,drop_years):#モデル関連に使用するdfの作成関数、３年分に展開して行う。(クラスタリングあり、モータ番号、艇番号なし)
    result_df=df
    result_df=result_df.drop(["racer_1_ID","racer_2_ID","racer_3_ID","racer_4_ID","racer_5_ID","racer_6_ID",],axis=1)#IDはいらないので削除
    result_df=result_df.replace(0.0000,{"racer_1_ave_st_time":0.22}).copy()#新人のave_st_timeを0.22に
    result_df=result_df.replace(0.0000,{"racer_2_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_3_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_4_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_5_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_6_ave_st_time":0.22}).copy()
    result_df=result_df.replace(0.0000,{"racer_1_doub_win":0.02}).copy()#新人の着に絡む確率ave_st_timeを0.02に(新人の半期の偏差から導出)
    result_df=result_df.replace(0.0000,{"racer_2_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_3_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_4_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_5_doub_win":0.02}).copy()
    result_df=result_df.replace(0.0000,{"racer_6_doub_win":0.02}).copy()

    year1=years[0]
    year2=years[1]
    year3=years[2]

    #ダミー変数化
    result_df_dummie=result_df.copy()
    race_dummie_df=pd.get_dummies(result_df_dummie['number_race'])#number_raceをダミー化
    for column, val in race_dummie_df.iteritems():
        result_df_dummie['race_{}'.format(int(column))]=val
    result_df_dummie=result_df_dummie.drop('number_race',axis=1).copy()

    cols=list(result_df_dummie.columns)
    male_cols=[s for s in cols if 'male' in s]#性別を示すカラムを取り出す

    #===========================新規、性別の取り出し機能が良くなかったため作り直す
    empty_arr=[0]*len(result_df_dummie)
    for col in male_cols:
        for number in np.arange(0,2,1):
              result_df_dummie['{}_{}'.format(col,int(number))]=empty_arr.copy()
        male_dummie_df=pd.get_dummies(result_df_dummie[col]).copy()#性別をダミー化
        for column, val in male_dummie_df.iteritems():
              result_df_dummie['{}_{}'.format(col,int(column))]=val.copy()
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()

    cols=list(result_df_dummie.columns)



    moter_cols=[s for s in cols if '_mo' in s]#モーター番号を示すカラムを取り出す
    boat_cols=[s for s in cols if '_bo' in s]#ボート番号を示すカラムを取り出す

    #boat、moterの情報は使わない、
    numbers=np.arange(1, 100, 1)
    empty_arr=[0]*len(result_df_dummie)
    for col in moter_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()
    for col in boat_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1).copy()

    #クラスタリング
    #分けてみるクラスタの数は[3,5,7,9]の4個
    #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    #学習データのdateを年に変換
    result_df_dummie['date']=pd.to_datetime(result_df_dummie['date'])#日付が文字列なのでdateを日付型に変換
    result_df_dummie['year']=result_df_dummie['date'].dt.year

    for year in drop_years:#削除したい年のデータを消す
        result_df_dummie=result_df_dummie[result_df_dummie['year']!=year]#削除対象の稔夫リストに入っている場合は削除
    #==========================================================================
    #result_df_dummie=result_df_dummie[result_df_dummie['year']!=2020].copy()#2020のデータを完全に切り離す。
    #==========================================================================

    #クラスタリングに邪魔だから消したいけど、後々使うものはいったんよけておく
    result=result_df_dummie['result_com'].values.copy()#
    money=result_df_dummie['money'].values.copy()#
    years=result_df_dummie['year'].values.copy()#

    #安全なところに移したら削除する
    result_df_dummie=result_df_dummie.drop('result_com',axis=1)
    result_df_dummie=result_df_dummie.drop('money',axis=1)
    result_df_dummie=result_df_dummie.drop('date',axis=1)
    #クラアスタリング用の学習、予測用のデータの切り分け
    clustar_final_test_df=result_df_dummie[(result_df_dummie['year']==year3)].copy()#2020のデータを最終チェックデータ(予測のターゲット)に。
    clustar_test_df = result_df_dummie[(result_df_dummie['year']==year1) | ((result_df_dummie['year']==year2) )].copy()#2018,2019のデータを検証用データに。
    clustar_train_df =  result_df_dummie[(result_df_dummie['year']!=year1) & (result_df_dummie['year']!=year2)& (result_df_dummie['year']!=year3) ].copy()#そのほかを学習データに

    #年の情報だけ切り分けに使ったからここで消す。
    clustar_final_test_df=clustar_final_test_df.drop('year',axis=1).copy()
    clustar_test_df=clustar_test_df.drop('year',axis=1).copy()
    clustar_train_df=clustar_train_df.drop('year',axis=1).copy()

    target_num_cluster=[3,5,7,9]#分けるクラスタ数によってモデルの名前を変える
    for num_cluster in target_num_cluster:
        Km = KMeans(random_state=7,n_clusters=num_cluster).fit(clustar_train_df)#rondom_stateはラッキーセブン
        final_test_pred =Km.predict(clustar_final_test_df)#rondom_stateはラッキーセブン
        test_pred =Km.predict(clustar_test_df)#rondom_stateはラッキーセブン
        train_pred = Km.predict(clustar_train_df)#rondom_stateはラッキーセブン
        #Km=========================実査に使うときはこれのモデルを会場ごとに保存して使用。
        clustar_final_test_df['num={}'.format(num_cluster)]=final_test_pred
        clustar_test_df['num={}'.format(num_cluster)]=test_pred
        clustar_train_df['num={}'.format(num_cluster)]=train_pred

    #結合して元の形に戻す。
    clustar_df=pd.concat([clustar_train_df, clustar_test_df,clustar_final_test_df]).copy()
    clustar_df['year']=years
    clustar_df['money']=money
    clustar_df['result_com']=result
    model_df=clustar_df.copy()
    return model_df



def model_score_rondom_3year_expand_select_year(version,years,place_name,result_df):#パラメータ探索関数(XGboostの出力を確率のやつを使用したバージョン、閾値の探索も行う。)3年間のデータを未知データとして扱い、精度予測モデル作成用の新形式のスコアシートを出力する
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    #model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'])#スコアを格納するdf

    year1=years[0]
    year2=years[1]
    year3=years[2]
    test_year1_df= result_df[(result_df['year']==year1)].copy()#2018のデータ
    test_year2_df= result_df[(result_df['year']==year2)].copy()#2018のデータ
    test_year3_df= result_df[(result_df['year']==year3)].copy()#2018のデータ

    train_df =  result_df[(result_df['year']!=year1) & (result_df['year']!=year2)& (result_df['year']!=year3) ].copy()#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    #final_test_df=final_test_df.drop(['year'],axis=1).copy()
    #test_df=test_df.drop(['year'],axis=1).copy()
    test_year1_df=test_year1_df.drop(['year'],axis=1).copy()
    test_year2_df=test_year2_df.drop(['year'],axis=1).copy()
    test_year3_df=test_year3_df.drop(['year'],axis=1).copy()

    train_df=train_df.drop(['year'],axis=1).copy()
    #金額の情報は横によけておく
    #final_test_money=pd.Series(final_test_df['money']).copy()
    #test_money=pd.Series(test_df['money']).copy()
    test_year1_money=pd.Series(test_year1_df['money']).copy()
    test_year2_money=pd.Series(test_year2_df['money']).copy()
    test_year3_money=pd.Series(test_year3_df['money']).copy()
    train_money=pd.Series(train_df['money']).copy()

    #出現数の分布
    result_com_s=train_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため

    for result_com_number in tqdm(result_com_df['result_com'].values):
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================
        result_s=result_com_df[result_com_df['result_com']==result_com]
        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_train_df=trans_result_com(result_com,result_train_df)

        result_test_year1_df=test_year1_df.copy()
        result_test_year1_df=trans_result_com(result_com,result_test_year1_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year2_df=test_year2_df.copy()
        result_test_year2_df=trans_result_com(result_com,result_test_year2_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year3_df=test_year3_df.copy()
        result_test_year3_df=trans_result_com(result_com,result_test_year3_df)#対象のラベルを１、それ以外を０に変換する関数

        result_train_df['money']=train_money
        result_test_year1_df['money']=test_year1_money
        result_test_year2_df['money']=test_year2_money
        result_test_year3_df['money']=test_year3_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,190)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        depths_arr=[5,8]

        for depth in depths_arr:#パラメータを可変していってスコアの探索を行う
            for sum_target_per in for_arr:

                index=sum_target_per-1
                target_per=100+sum_target_per#学習データを増やす
                target_per_arr[index]=target_per
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df.copy()#ベースのデータフレームをコピー
                target_1_df=target_df[target_df['result_com']==1]
                len_1=len(target_1_df)
                target_0_df=target_df[target_df['result_com']==0]
                len_0=len(target_0_df)
                target_0_df=target_0_df.iloc[(len_0-int(len_1*(target_per/100))):len_0].copy()#1に対する目標の割合ぶん0の結果だったレースを抽出（後ろから抽出）
                target_train_df=pd.concat([target_1_df, target_0_df]).copy()
                #学習＆予測ぱーと========================================================================
                #==========================================================================================================================================
                #データの切り分け
                target_x_train=target_train_df.drop('money',axis=1).copy()
                target_x_train=target_x_train.drop('result_com',axis=1)

                target_x_year1_test=result_test_year1_df.drop('money',axis=1).copy()
                target_x_year1_test=target_x_year1_test.drop('result_com',axis=1)

                target_x_year2_test=result_test_year2_df.drop('money',axis=1).copy()
                target_x_year2_test=target_x_year2_test.drop('result_com',axis=1)

                target_x_year3_test=result_test_year3_df.drop('money',axis=1).copy()
                target_x_year3_test=target_x_year3_test.drop('result_com',axis=1)

                target_y_train=target_train_df['result_com'].copy()
                target_y_year1_test=result_test_year1_df['result_com'].copy()
                target_y_year2_test=result_test_year2_df['result_com'].copy()
                target_y_year3_test=result_test_year3_df['result_com'].copy()


                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=False)#学習データ内でさらに分割してロスをもとに修正をする。

                #XGboostのデータ型に変換する
                # train = xgb.DMatrix(train_x, label=train_y)#学習用
                # valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                # #test = xgb.DMatrix(target_x_test)#実際に使った時の利益率の算出用
                # #final_test = xgb.DMatrix(target_x_final_test)#実際に使った時の利益率の算出用
                # year1 = xgb.DMatrix(target_x_year1_test)#学習時のロス修正用
                # year2 = xgb.DMatrix(target_x_year2_test)#学習時のロス修正用
                # year3 = xgb.DMatrix(target_x_year3_test)#学習時のロス修正用
                #xgb.config_context(verbosity=0)
                # param = {'max_depth': depth, #パラメータの設定
                #                  'eta': 0.3,
                #                  #'objective': 'binary:hinge',
                #                  'objective': 'binary:logistic',#確率で出力
                #                  'eval_metric': 'logloss',
                #                  'verbosity':0,
                #                  'subsample':0.8,
                #                  'nthread':10,
                #                  'gpu_id':0,
                #                  'seed':7,
                #                  'tree_method':'gpu_hist'
                #                 }
                # evallist = [(train, 'train'),(valid, 'eval')]#学習時にバリデーションを監視するデータの指定。
                # num_round = 800
                #bst = xgb.train(param, train,num_round,evals=evallist, early_stopping_rounds=30, verbose_eval=0 )
                clf=RandomForestClassifier(random_state=7,n_estimators=1000,max_depth=depth,n_jobs=10)
                clf=clf.fit(target_x_train,target_y_train)

                # 未知データに対する予測値
                #predict_y_year1_test=bst.predict(year1)
                #predict_y_year1_test=[arr[1] for arr in clf.predict_proba(target_x_year1_test)]
                predict_y_year1_test=clf.predict(target_x_year1_test)
                #predict_y_year2_test=bst.predict(year2)
                #predict_y_year2_test=[arr[1] for arr in clf.predict_proba(target_x_year2_test)]
                predict_y_year2_test=clf.predict(target_x_year2_test)
                #predict_y_year3_test=bst.predict(year3)
                #predict_y_year3_test=[arr[1] for arr in clf.predict_proba(target_x_year3_test)]
                predict_y_year3_test=clf.predict(target_x_year3_test)
                #==========================================================================================================================================
                #[1]の正答率を見る
                # pred_year1_test_df=pd.DataFrame({'pred_proba':predict_y_year1_test#確率分布での出力
                #                                  , 'trans_result':target_y_year1_test})
                # pred_year2_test_df=pd.DataFrame({'pred_proba':predict_y_year2_test#確率分布での出力
                #                                  , 'trans_result':target_y_year2_test})
                # pred_year3_test_df=pd.DataFrame({'pred_proba':predict_y_year3_test#確率分布での出力
                #                                  , 'trans_result':target_y_year3_test})
                pred_year1_test_df=pd.DataFrame({'pred':predict_y_year1_test#確率分布での出力
                                                 , 'trans_result':target_y_year1_test})
                pred_year2_test_df=pd.DataFrame({'pred':predict_y_year2_test#確率分布での出力
                                                 , 'trans_result':target_y_year2_test})
                pred_year3_test_df=pd.DataFrame({'pred':predict_y_year3_test#確率分布での出力
                                                 , 'trans_result':target_y_year3_test})

                #th_arr=[0.85,0.9,0.92]
                #for th in th_arr:
                #trans_df=pred_th_trans(pred_test_df,th)
                #閾値をもとに予測を変換
                # year1_trans_df=pred_th_trans(pred_year1_test_df,th)
                # year2_trans_df=pred_th_trans(pred_year2_test_df,th)
                # year3_trans_df=pred_th_trans(pred_year3_test_df,th)
                year1_trans_df=pred_year1_test_df.copy()#閾値での変換は行わない
                year2_trans_df=pred_year2_test_df.copy()#閾値での変換は行わない
                year3_trans_df=pred_year3_test_df.copy()#閾値での変換は行わない
                #/////収益計算の項
                year1_trans_df['money']=test_year1_money
                year1_trans_df['true_result']=test_year1_df['result_com']
                year2_trans_df['money']=test_year2_money
                year2_trans_df['true_result']=test_year2_df['result_com']
                year3_trans_df['money']=test_year3_money
                year3_trans_df['true_result']=test_year3_df['result_com']
                #配当金の情報も考慮する。
                #result_gain_base_df=calc_gain(trans_df)
                year1_result_gain_base_df=calc_gain(year1_trans_df)
                year2_result_gain_base_df=calc_gain(year2_trans_df)
                year3_result_gain_base_df=calc_gain(year3_trans_df)

                #scoreのseriesに情報書き込み==================
                #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                #model_score_s=pd.Series(dtype='float64')
                #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
                model_score_s=pd.Series(index=['target_com','depth','target_per','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')

                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #model_score_s['threshold']=th

                result_gain_df_arr=[year1_result_gain_base_df,year2_result_gain_base_df,year3_result_gain_base_df]
                year_labels=[1,2,3]
                #年のごとのスコア情報を横に展開していく
                for year_df,label in zip(result_gain_df_arr,year_labels):
                    if label !=3:
                        #model_score_s['総収益']=result_gain_base_df["gain"].sum()
                        model_score_s['total_get_year{year}'.format(year=label)]=year_df["gain"].sum()
                        #model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
                        model_score_s['total_use_year{year}'.format(year=label)]=100*year_df["pred"].sum()
                        #model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
                        model_score_s['num_com_year{year}'.format(year=label)]=year_df['trans_result'].sum()
                        #model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
                        model_score_s['num_pred_year{year}'.format(year=label)]=year_df['pred'].sum()
                        #model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                        model_score_s['gain_year{year}'.format(year=label)]=(model_score_s['total_get_year{year}'.format(year=label)]/model_score_s['total_use_year{year}'.format(year=label)])*100
                        #model_score_s['的中数']=result_gain_base_df['hit'].sum()
                        model_score_s['num_hit_year{year}'.format(year=label)]=year_df['hit'].sum()
                        #model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
                        model_score_s['buy_hit_per_year{year}'.format(year=label)]=(model_score_s['num_hit_year{year}'.format(year=label)]/ model_score_s['num_pred_year{year}'.format(year=label)])*100
                    else:
                        model_score_s['gain_year{year}'.format(year=label)]=(year_df["gain"].sum()/(100*year_df["pred"].sum()))*100
                model_score_df=model_score_df.append(model_score_s,ignore_index=True,sort=False)
    #モデルの「スコアを保存
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_3year_expand_{V}_until_{latest}.csv".format(place_name=place_name,V=version,latest=years[0])#作成したデータの書き込み先
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None


def model_score_XGboost_3year_expand_select_year(version,years,place_name,result_df):#パラメータ探索関数(XGboostの出力を確率のやつを使用したバージョン、閾値の探索も行う。)3年間のデータを未知データとして扱い、精度予測モデル作成用の新形式のスコアシートを出力する
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    #model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'])#スコアを格納するdf

    year1=years[0]
    year2=years[1]
    year3=years[2]
    test_year1_df= result_df[(result_df['year']==year1)].copy()#2018のデータ
    test_year2_df= result_df[(result_df['year']==year2)].copy()#2018のデータ
    test_year3_df= result_df[(result_df['year']==year3)].copy()#2018のデータ

    train_df =  result_df[(result_df['year']!=year1) & (result_df['year']!=year2)& (result_df['year']!=year3) ].copy()#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    #final_test_df=final_test_df.drop(['year'],axis=1).copy()
    #test_df=test_df.drop(['year'],axis=1).copy()
    test_year1_df=test_year1_df.drop(['year'],axis=1).copy()
    test_year2_df=test_year2_df.drop(['year'],axis=1).copy()
    test_year3_df=test_year3_df.drop(['year'],axis=1).copy()

    train_df=train_df.drop(['year'],axis=1).copy()
    #金額の情報は横によけておく
    #final_test_money=pd.Series(final_test_df['money']).copy()
    #test_money=pd.Series(test_df['money']).copy()
    test_year1_money=pd.Series(test_year1_df['money']).copy()
    test_year2_money=pd.Series(test_year2_df['money']).copy()
    test_year3_money=pd.Series(test_year3_df['money']).copy()
    train_money=pd.Series(train_df['money']).copy()

    #出現数の分布
    result_com_s=train_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため

    for result_com_number in tqdm(result_com_df['result_com'].values):
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================
        result_s=result_com_df[result_com_df['result_com']==result_com]
        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_train_df=trans_result_com(result_com,result_train_df)

        result_test_year1_df=test_year1_df.copy()
        result_test_year1_df=trans_result_com(result_com,result_test_year1_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year2_df=test_year2_df.copy()
        result_test_year2_df=trans_result_com(result_com,result_test_year2_df)#対象のラベルを１、それ以外を０に変換する関数
        result_test_year3_df=test_year3_df.copy()
        result_test_year3_df=trans_result_com(result_com,result_test_year3_df)#対象のラベルを１、それ以外を０に変換する関数

        result_train_df['money']=train_money
        result_test_year1_df['money']=test_year1_money
        result_test_year2_df['money']=test_year2_money
        result_test_year3_df['money']=test_year3_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,190)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        depths_arr=[5,8]

        for depth in depths_arr:#パラメータを可変していってスコアの探索を行う
            for sum_target_per in for_arr:

                index=sum_target_per-1
                target_per=100+sum_target_per#学習データを増やす
                target_per_arr[index]=target_per
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df.copy()#ベースのデータフレームをコピー
                target_1_df=target_df[target_df['result_com']==1]
                len_1=len(target_1_df)
                target_0_df=target_df[target_df['result_com']==0]
                len_0=len(target_0_df)
                target_0_df=target_0_df.iloc[(len_0-int(len_1*(target_per/100))):len_0].copy()#1に対する目標の割合ぶん0の結果だったレースを抽出（後ろから抽出）
                target_train_df=pd.concat([target_1_df, target_0_df]).copy()
                #学習＆予測ぱーと========================================================================
                #==========================================================================================================================================
                #データの切り分け
                target_x_train=target_train_df.drop('money',axis=1).copy()
                target_x_train=target_x_train.drop('result_com',axis=1)

                target_x_year1_test=result_test_year1_df.drop('money',axis=1).copy()
                target_x_year1_test=target_x_year1_test.drop('result_com',axis=1)

                target_x_year2_test=result_test_year2_df.drop('money',axis=1).copy()
                target_x_year2_test=target_x_year2_test.drop('result_com',axis=1)

                target_x_year3_test=result_test_year3_df.drop('money',axis=1).copy()
                target_x_year3_test=target_x_year3_test.drop('result_com',axis=1)

                target_y_train=target_train_df['result_com'].copy()
                target_y_year1_test=result_test_year1_df['result_com'].copy()
                target_y_year2_test=result_test_year2_df['result_com'].copy()
                target_y_year3_test=result_test_year3_df['result_com'].copy()


                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=False)#学習データ内でさらに分割してロスをもとに修正をする。

                #XGboostのデータ型に変換する
                train = xgb.DMatrix(train_x, label=train_y)#学習用
                valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                #test = xgb.DMatrix(target_x_test)#実際に使った時の利益率の算出用
                #final_test = xgb.DMatrix(target_x_final_test)#実際に使った時の利益率の算出用
                year1 = xgb.DMatrix(target_x_year1_test)#学習時のロス修正用
                year2 = xgb.DMatrix(target_x_year2_test)#学習時のロス修正用
                year3 = xgb.DMatrix(target_x_year3_test)#学習時のロス修正用
                #xgb.config_context(verbosity=0)
                param = {'max_depth': depth, #パラメータの設定
                                 'eta': 0.3,
                                 #'objective': 'binary:hinge',
                                 'objective': 'binary:logistic',#確率で出力
                                 'eval_metric': 'logloss',
                                 'verbosity':0,
                                 'subsample':0.8,
                                 'nthread':10,
                                 'gpu_id':0,
                                 'seed':7,
                                 'tree_method':'gpu_hist'
                                }
                evallist = [(train, 'train'),(valid, 'eval')]#学習時にバリデーションを監視するデータの指定。
                num_round = 800
                bst = xgb.train(param, train,num_round,evals=evallist, early_stopping_rounds=30, verbose_eval=0 )

                # 未知データに対する予測値
                #predict_y_test=bst.predict(test)
                predict_y_year1_test=bst.predict(year1)
                predict_y_year2_test=bst.predict(year2)
                predict_y_year3_test=bst.predict(year3)
                #==========================================================================================================================================
                #[1]の正答率を見る
                #pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                #                          , 'trans_result':target_y_test})
                pred_year1_test_df=pd.DataFrame({'pred_proba':predict_y_year1_test#確率分布での出力
                                                 , 'trans_result':target_y_year1_test})
                pred_year2_test_df=pd.DataFrame({'pred_proba':predict_y_year2_test#確率分布での出力
                                                 , 'trans_result':target_y_year2_test})
                pred_year3_test_df=pd.DataFrame({'pred_proba':predict_y_year3_test#確率分布での出力
                                                 , 'trans_result':target_y_year3_test})

                th_arr=[0.85,0.9,0.92]
                for th in th_arr:
                    #trans_df=pred_th_trans(pred_test_df,th)
                    #閾値をもとに予測を変換
                    year1_trans_df=pred_th_trans(pred_year1_test_df,th)
                    year2_trans_df=pred_th_trans(pred_year2_test_df,th)
                    year3_trans_df=pred_th_trans(pred_year3_test_df,th)
                    count=0
                    #/////収益計算の項
                    year1_trans_df['money']=test_year1_money
                    year1_trans_df['true_result']=test_year1_df['result_com']
                    year2_trans_df['money']=test_year2_money
                    year2_trans_df['true_result']=test_year2_df['result_com']
                    year3_trans_df['money']=test_year3_money
                    year3_trans_df['true_result']=test_year3_df['result_com']
                    #配当金の情報も考慮する。
                    #result_gain_base_df=calc_gain(trans_df)
                    year1_result_gain_base_df=calc_gain(year1_trans_df)
                    year2_result_gain_base_df=calc_gain(year2_trans_df)
                    year3_result_gain_base_df=calc_gain(year3_trans_df)



                    #scoreのseriesに情報書き込み==================
                    #model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                    #model_score_s=pd.Series(dtype='float64')
                    model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
                    model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                    model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                    model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                    model_score_s['threshold']=th

                    result_gain_df_arr=[year1_result_gain_base_df,year2_result_gain_base_df,year3_result_gain_base_df]
                    year_labels=[1,2,3]
                    #年のごとのスコア情報を横に展開していく
                    for year_df,label in zip(result_gain_df_arr,year_labels):
                        if label !=3:
                            #model_score_s['総収益']=result_gain_base_df["gain"].sum()
                            model_score_s['total_get_year{year}'.format(year=label)]=year_df["gain"].sum()
                            #model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
                            model_score_s['total_use_year{year}'.format(year=label)]=100*year_df["pred"].sum()
                            #model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
                            model_score_s['num_com_year{year}'.format(year=label)]=year_df['trans_result'].sum()
                            #model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
                            model_score_s['num_pred_year{year}'.format(year=label)]=year_df['pred'].sum()
                            #model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                            model_score_s['gain_year{year}'.format(year=label)]=(model_score_s['total_get_year{year}'.format(year=label)]/model_score_s['total_use_year{year}'.format(year=label)])*100
                            #model_score_s['的中数']=result_gain_base_df['hit'].sum()
                            model_score_s['num_hit_year{year}'.format(year=label)]=year_df['hit'].sum()
                            #model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
                            model_score_s['buy_hit_per_year{year}'.format(year=label)]=(model_score_s['num_hit_year{year}'.format(year=label)]/ model_score_s['num_pred_year{year}'.format(year=label)])*100
                        else:
                            model_score_s['gain_year{year}'.format(year=label)]=(year_df["gain"].sum()/(100*year_df["pred"].sum()))*100
                    model_score_df=model_score_df.append(model_score_s,ignore_index=True,sort=False)
    #モデルの「スコアを保存
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_3year_expand_{V}_until_{latest}.csv".format(place_name=place_name,V=version,latest=years[0])#作成したデータの書き込み先
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None

#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
#実際に実行をする関数、上のまとめ役==================================================================================================================================================================================================================
def version_3_1_3years(version,years,place_name,base_df):#閾値で予測を変えるバージョンのxgboost版
    result_df=data_making_clustar_3year_expand(base_df,years)
    model_score_XGboost_3year_expand(version,years,place_name,result_df)#閾値を決めて変換するver

def version_3_1(version,years,place_name,base_df):#閾値で予測を変えるバージョンのrandom_forest版
    result_df=data_making_clustar(base_df,years)
    model_score_XGboost(version,years,place_name,result_df)#閾値を決めて変換するver




def version_3_2_3years(version,years,place_name,base_df):#閾値で予測を変えるバージョンのrandom_forest版
    result_df=data_making_clustar_3year_expand(base_df,years)
    model_score_rondom_3year_expand(version,years,place_name,result_df)#閾値を決めて変換するver

def version_3_2(version,years,place_name,base_df):#閾値で予測を変えるバージョンのrandom_forest版
    result_df=data_making_clustar(base_df,years)
    model_score_rondom(version,years,place_name,result_df)#閾値を決めて変換するver
