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
from sklearn.preprocessing import StandardScaler#モデルの評価用に標準化する関数
import scipy.stats#モデルの評価用に標準化する関数

#関数内で使う関数=========================================================================================================================
#関数内で使う関数=========================================================================================================================
#関数内で使う関数=========================================================================================================================
#関数内で使う関数=========================================================================================================================
#関数内で使う関数=========================================================================================================================
#関数内で使う関数=========================================================================================================================


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

def add_result_class(model,score_df):#モデルでの予測結果を学習時に使用したパラメータシートに追加して返す関数
    #out_figはオプション、グラフの出力の有り無し
    score_sheet_base=score_df.copy()
    score_sheet_shuffle=score_sheet_base.sample(frac=1, random_state=7)#行をシャッフル
    num_data=len(score_sheet_shuffle)#データの件数

    #前処理    これはクラス分類なので閾値で最新収益を1,0に変換する
    score_sheet_shuffle['gain_label']=0#実際の収益の有無を確認(110を利益有り無しの閾値とする )
    score_sheet_shuffle.loc[score_sheet_shuffle['gain_year3'] < 110, 'gain_label'] =0
    score_sheet_shuffle.loc[score_sheet_shuffle['gain_year3'] >= 110, 'gain_label'] =1
    ex_df=score_sheet_shuffle.copy()#gain_year3の切り抜き用
    score_sheet_shuffle=score_sheet_shuffle.drop(['gain_year3'],axis=1).copy()

    #データの分割========================================================================================================
    test_df = score_sheet_shuffle[int(num_data*0.8):].copy()#20%のデータを検証用データに。
    train_df =  score_sheet_shuffle[:int(num_data*0.8)].copy()#そのほかを学習データに
    gain_s=ex_df[int(num_data*0.8):]['gain_year3'].copy()#検証用データの実収益部分を切り抜く。

    test_x=test_df.drop(['gain_label'],axis=1).copy()
    train_x=train_df.drop(['gain_label'],axis=1).copy()
    test_y=test_df['gain_label']
    train_y=train_df['gain_label']
    test_pred_df=test_df.copy()#予測の結合先を作っておく
    train_pred_df=train_df.copy()#予測の結合先を作っておく

    #予測を出力(確率分布)
    train_pred_proba_arr=model.predict_proba(train_x)
    train_pred_proba=[arr[1] for arr in train_pred_proba_arr]#確率分布の二次元配列なので[1]の部分だけ取り出す
    test_pred_proba_arr = model.predict_proba(test_x)
    test_pred_proba=[arr[1] for arr in test_pred_proba_arr]#確率分布の二次元配列なので[1]の部分だけ取り出す
    test_pred_df["pred_proba"]=test_pred_proba#予測を結合
    train_pred_df["pred_proba"]=train_pred_proba#予測を結合

    #確率分布の閾値で予測をバイナリに変換(仮)
    #test_pred_df=pred_th_trans(test_pred_df,0.5)
    #train_pred_df=pred_th_trans(train_pred_df,0.5)

#     #モデルのスコアの計算
#     col1='pred'
#     col2='gain_label'
#     cross_df,train_precision=making_cross(col1,col2 ,train_pred_df)
#     cross_df,test_precision=making_cross(col1,col2 ,test_pred_df)
#     print("train_precision:{}__test_precision:{}".format(train_precision,test_precision))

    test_pred_df['gain_year3']=gain_s#実際の収益の情報を追加する
    return test_pred_df

def making_cross(col1,col2,base_df,out_cross=0):#クロス収益図を作成してpurecisionを算出する関数
    cross_df=pd.DataFrame(columns=["{}_1".format(col2),"{}_0".format(col2),'sum']
                         ,index=["{}_1".format(col1),"{}_0".format(col1),'sum'])#クロス集計の結果の格納df
    cross_df.at["{}_1".format(col1), "{}_1".format(col2)]=len(base_df[(base_df[col1]==1) & (base_df[col2]==1)])#左上
    cross_df.at["{}_1".format(col1), "{}_0".format(col2)]=len(base_df[(base_df[col1]==1) & (base_df[col2]==0)])#右上
    cross_df.at["{}_0".format(col1), "{}_1".format(col2)]=len(base_df[(base_df[col1]==0) & (base_df[col2]==1)])#左下
    cross_df.at["{}_0".format(col1), "{}_0".format(col2)]=len(base_df[(base_df[col1]==0) & (base_df[col2]==0)])#右下

    cross_df.at["{}_1".format(col1), "sum"]=cross_df.at["{}_1".format(col1), "{}_1".format(col2)]+cross_df.at["{}_1".format(col1), "{}_0".format(col2)]
    cross_df.at["{}_0".format(col1), "sum"]=cross_df.at["{}_0".format(col1), "{}_1".format(col2)]+cross_df.at["{}_0".format(col1), "{}_0".format(col2)]
    cross_df.at["sum", "{}_1".format(col2)]=cross_df.at["{}_1".format(col1), "{}_1".format(col2)]+cross_df.at["{}_0".format(col1), "{}_1".format(col2)]
    cross_df.at["sum", "{}_0".format(col2)]=cross_df.at["{}_1".format(col1), "{}_0".format(col2)]+cross_df.at["{}_0".format(col1), "{}_0".format(col2)]
    if out_cross==1:
        display(cross_df)
    #precisionを算出
    try:
        #precision=(cross_df.at["{}_1".format(col1), "{}_1".format(col2)]/(cross_df.at["{}_1".format(col1), "{}_1".format(col2)]+cross_df.at["{}_1".format(col1), "{}_0".format(col2)]))*100
        precision=(cross_df.at["{}_1".format(col1), "{}_1".format(col2)]/(cross_df.at["{}_1".format(col1),'sum']))*100
    except ZeroDivisionError:
        precision=0

    return cross_df,precision

#モデルの選定を行う際に使う関数=========================================================================================================================
#モデルの選定を行う際に使う関数=========================================================================================================================
#モデルの選定を行う際に使う関数=========================================================================================================================
#モデルの選定を行う際に使う関数=========================================================================================================================
#モデルの選定を行う際に使う関数=========================================================================================================================


def model_selection_save(expand_score_df,score_df,place_name,version,th=0.8):#三年分のスコアシートを与えたら[モデル選定モデル]を作成して保存、かつ実際に使用する購買予測モデルのスコアシートを吐き出してくれる関数(use_model)(モデルの保存も同時に行う。)
    #expand_score_df:::モデル選定のためのモデル作成の学習データ用の三年間を横に展開したスコアシート
    #score_df:::実際にも本番で使うモデルを選ぶためのシート（答えなし）
    score_sheet_base=expand_score_df.copy()
    score_sheet_shuffle=score_sheet_base.sample(frac=1, random_state=7)#行をシャッフル
    num_data=len(score_sheet_shuffle)#データの件数

    #前処理    これはクラス分類なので閾値で最新収益を1,0に変換する
    score_sheet_shuffle['gain_label']=0#実際の収益の有無を確認(110を利益有り無しの閾値とする )
    score_sheet_shuffle.loc[score_sheet_shuffle['gain_year3'] < 110, 'gain_label'] =0
    score_sheet_shuffle.loc[score_sheet_shuffle['gain_year3'] >= 110, 'gain_label'] =1
    ex_df=score_sheet_shuffle.copy()#gain_year3の切り抜き用
    score_sheet_shuffle=score_sheet_shuffle.drop(['gain_year3'],axis=1).copy()
    #選定モデルの学習、保存パート============================================================================================================================================================================
    #選定モデルの学習、保存パート============================================================================================================================================================================
    #選定モデルの学習、保存パート============================================================================================================================================================================
    #選定モデルの学習、保存パート============================================================================================================================================================================
    #選定モデルの学習、保存パート============================================================================================================================================================================

    #データの分割========================================================================================================
    test_df = score_sheet_shuffle[int(num_data*0.8):].copy()#20%のデータを検証用データに。
    train_df =  score_sheet_shuffle[:int(num_data*0.8)].copy()#そのほかを学習データに
    gain_s=ex_df[int(num_data*0.8):]['gain_year3'].copy()#検証用データの実収益部分を切り抜く。

    test_x=test_df.drop(['gain_label'],axis=1).copy()
    train_x=train_df.drop(['gain_label'],axis=1).copy()
    test_y=test_df['gain_label']
    train_y=train_df['gain_label']
    test_pred_df=test_df.copy()#予測の結合先を作っておく
    train_pred_df=train_df.copy()#予測の結合先を作っておく

    #ざっと学習
    rc = RandomForestClassifier(n_jobs=8, random_state=7,n_estimators=100,max_depth=10)
    rc.fit(train_x,train_y)
    pickle_path="../../bot_database/{place_name}/model_pickle_{place_name}/model_selection_{place_name}_{V}.sav".format(place_name=place_name,V=version)#モデルを保存
    #pickle_path="check_selection.sav"
    pickle.dump(rc, open(pickle_path, "wb"))#モデルの保存
    clf=pickle.load(open(pickle_path, 'rb'))#モデルを格納読み込む

    #予測を出力(確率分布)
    train_pred_proba_arr=clf.predict_proba(train_x)
    train_pred_proba=[arr[1] for arr in train_pred_proba_arr]#確率分布の二次元配列なので[1]の部分だけ取り出す
    test_pred_proba_arr = clf.predict_proba(test_x)
    test_pred_proba=[arr[1] for arr in test_pred_proba_arr]#確率分布の二次元配列なので[1]の部分だけ取り出す
    test_pred_df["pred_proba"]=test_pred_proba#予測を結合
    train_pred_df["pred_proba"]=train_pred_proba#予測を結合

    #確率分布の閾値で予測をバイナリに変換(閾値はいったん決めで0.8にする(引数で簡単に変えられるからよろしくうううう))
    test_pred_df=pred_th_trans(test_pred_df,th)
    train_pred_df=pred_th_trans(train_pred_df,th)
    #モデルのスコアの計算
    train_score=rc.score(train_x, train_y)
    test_score=rc.score(test_x, test_y)
    print("train:{}__test:{}".format(train_score,test_score))

    col1='pred'
    col2='gain_label'
    cross_df,train_precision=making_cross(col1,col2 ,train_pred_df)
    cross_df,test_precision=making_cross(col1,col2 ,test_pred_df)
    print("CHECK::::train_precision:{}__test_precision:{}".format(train_precision,test_precision))#精度も一応算出しておく（閾値は0.8）

    #使うモデルの選定を行う。
    use_model_check_df=pd.DataFrame(columns=test_pred_df.columns)#最終的に使うと判定されたモデルのパラメータを格納するDF
    selection_df=test_pred_df.copy()
    selection_df['gain_year3']=gain_s#実際の収益の情報を追加する
    selection_df=selection_df[selection_df['pred_proba']>=th].copy()#閾値はいったん決めで区切る
    candidate_com=selection_df['target_com'].value_counts().index#閾値を超え、利益が出やすいと判断されたパラメータのあるcomを重複なく抜き出す
    for com in candidate_com:
        com_selection_df=selection_df[selection_df['target_com']==com].copy()
        com_selection_df=com_selection_df.sort_values('pred_proba', ascending=False).iloc[:1]#各組の一番probaが高かったものを残す。
        use_model_check_df=pd.concat([use_model_check_df, com_selection_df], axis=0)


    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================
    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================
    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================
    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================
    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================
    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================
    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================
    #選定モデルを使っての本番で使うパラメータの選定パート============================================================================================================================================================================

    #expand_score_df:::モデル選定のためのモデル作成の学習データ用の三年間を横に展開したスコアシート
    #score_df:::実際にも本番で使うモデルを選ぶためのシート（答えなし）
    has_proba_score_df=score_df.copy()
    #予測を出力(確率分布)
    pred_proba_arr = clf.predict_proba(score_df)
    pred_proba=[arr[1] for arr in pred_proba_arr]#確率分布の二次元配列なので[1]の部分だけ取り出す
    has_proba_score_df["pred_proba"]=pred_proba#予測を結合

    #確率分布の閾値で予測をバイナリに変換(閾値はいったん決めで0.8にする(引数で簡単に変えられるからよろしくうううう))
    has_proba_score_df=pred_th_trans(has_proba_score_df,th)

    #使うモデルの選定を行う。
    use_model_df=pd.DataFrame(columns=has_proba_score_df.columns)#最終的に使うと判定されたモデルのパラメータを格納するDF
    selection_df=has_proba_score_df.copy()
    selection_df=selection_df[selection_df['pred_proba']>=th].copy()#閾値はいったん決めで区切る
    candidate_com=selection_df['target_com'].value_counts().index#閾値を超え、利益が出やすいと判断されたパラメータのあるcomを重複なく抜き出す
    for com in candidate_com:
        com_selection_df=selection_df[selection_df['target_com']==com].copy()
        com_selection_df=com_selection_df.sort_values('pred_proba', ascending=False).iloc[:1]#各組の一番probaが高かったものを残す。
        use_model_df=pd.concat([use_model_df, com_selection_df], axis=0)
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/use_model/use_model_{place_name}_{V}.csv".format(place_name=place_name,V=version)#選定されたモデルのリストを出力
    use_model_df.to_csv(dir_path, encoding='utf_8_sig')


    return use_model_check_df,use_model_df#決まったパラメータでモデルを作成できてるかのチェック

def save_clustar_model(result_base_df,place_name,version):#クラスタリングあり、モータ番号、艇番号なし
    result_df=result_base_df
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

    #boat、moterの情報は使わない、
    numbers=np.arange(1, 100, 1)
    empty_arr=[0]*len(result_df_dummie)
    for col in moter_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1)
    for col in boat_cols:
        result_df_dummie=result_df_dummie.drop('{}'.format(col),axis=1)

    #クラスタリング
    #分けてみるクラスタの数は[3,5,7,9]の4個
    #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    #学習データのdateを年に変換
    result_df_dummie['date']=pd.to_datetime(result_df_dummie['date'])#日付が文字列なのでdateを日付型に変換
    result_df_dummie['year']=result_df_dummie['date'].dt.year

    #クラスタリングに邪魔だから消したいけど、後々使うものはいったんよけておく
    result=result_df_dummie['result_com'].values#
    money=result_df_dummie['money'].values#
    years=result_df_dummie['year'].values#

    #安全なところに移したら削除する
    result_df_dummie=result_df_dummie.drop('result_com',axis=1)
    result_df_dummie=result_df_dummie.drop('money',axis=1)
    result_df_dummie=result_df_dummie.drop('date',axis=1)

    #クラアスタリング用の学習、予測用のデータの切り分け
    clustar_test_df = result_df_dummie[(result_df_dummie['year']==2019) | ((result_df_dummie['year']==2020) )].copy()#2019,2020のデータを検証用データに。
    clustar_train_df =  result_df_dummie[(result_df_dummie['year']!=2019) & ((result_df_dummie['year']!=2020) )].copy()#そのほかを学習データに

    #年の情報だけ切り分けに使ったからここで消す。
    clustar_test_df=clustar_test_df.drop('year',axis=1)
    clustar_train_df=clustar_train_df.drop('year',axis=1)

    target_num_cluster=[3,5,7,9]#分けるクラスタ数によってモデルの名前を変える
    for num_cluster in target_num_cluster:
        Km = KMeans(random_state=7,n_clusters=num_cluster).fit(clustar_train_df)#rondom_stateはラッキーセブン
        train_pred = Km.predict(clustar_train_df)#rondom_stateはラッキーセブン
        test_pred =Km.predict(clustar_test_df)#rondom_stateはラッキーセブン
        pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/clustering_{place_name}_num_{num_cluster}_{V}.sav".format(place_name=place_name,num_cluster=num_cluster,V=version)#モデルを保存
        pickle.dump(Km, open(pickle_path, "wb"))#モデルの保存
        clustar_train_df['num={}'.format(num_cluster)]=train_pred
        clustar_test_df['num={}'.format(num_cluster)]=test_pred

def save_model_XGboost_V3_1(result_base_df,use_model_df,place_name,version):
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'])#スコアを格納するdf
    year1=years[0]
    year2=years[1]
    test_year1_df= result_df[(result_df['year']==year1)].copy()#2019のデータ
    test_year2_df= result_df[(result_df['year']==year2)].copy()#2020のデータ

    train_df =  result_df[(result_df['year']!=year1) & (result_df['year']!=year2)].copy()#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    test_year1_df=test_year1_df.drop(['year'],axis=1).copy()
    test_year2_df=test_year2_df.drop(['year'],axis=1).copy()

    train_df=train_df.drop(['year'],axis=1).copy()
    #金額の情報は横によけておく
    test_year1_money=pd.Series(test_year1_df['money']).copy()
    test_year2_money=pd.Series(test_year2_df['money']).copy()
    train_money=pd.Series(train_df['money']).copy()

    #出現数の分布
#     result_com_s=train_df['result_com'].value_counts()
#     result_com_s=result_com_s.sort_index()
#     result_com_df=pd.DataFrame({'result_com':result_com_s.index})
#     result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため

    for index, model_row in use_model_df.iterrows():
                #パラメータ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        #======================================================================================
        #result_com=int(model_row['target_com'])
        result_com=int(model_row['target_com'])
        depth=int(model_row['depth'])
        target_per=int(model_row['target_per'])
        th=float(model_row['threshold'])
        #======================================================================================
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
        clf = xgb.train(param, train,num_round,evals=evallist, early_stopping_rounds=30, verbose_eval=0 )
        #==========================================================================================================================================
        #==========================================================================================================================================

        pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/com{com}_{depth}_{target_per}_{th}_{place_name}.sav".format(place_name=place_name,com=result_com,depth=depth,target_per=target_per,th=th)#モデルを保存
        pickle.dump(clf, open(pickle_path, "wb"))#モデルの保存
        #その場でpickleの方を読み込んでpickleの出力の方を確認する
        bst=pickle.load(open(pickle_path, 'rb'))#組番号に対応したモデルを格納#モデルの読み込み

        #==========================================================================================================================================
        #==========================================================================================================================================
        # 未知データに対する予測値
        predict_y_year1_test=bst.predict(year1)
        predict_y_year2_test=bst.predict(year2)
        #==========================================================================================================================================
        #[1]の正答率を見る
        pred_year1_test_df=pd.DataFrame({'pred_proba':predict_y_year1_test#確率分布での出力
                                         , 'trans_result':target_y_year1_test})
        pred_year2_test_df=pd.DataFrame({'pred_proba':predict_y_year2_test#確率分布での出力
                                         , 'trans_result':target_y_year2_test})

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
        model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','total_get_year1', 'total_use_year1','num_com_year1','num_pred_year1','num_hit_year1','buy_hit_per_year1','gain_year1','total_get_year2', 'total_use_year2','num_com_year2','num_pred_year2','num_hit_year2','buy_hit_per_year2','gain_year2','gain_year3'],dtype='float64')
        model_score_s['target_com']=result_com#目標としているresult_comラベル番号
        model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
        model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
        model_score_s['threshold']=th

        result_gain_df_arr=[year1_result_gain_base_df,year2_result_gain_base_df]
        year_labels=[1,2]
        #年のごとのスコア情報を横に展開していく
        for year_df,label in zip(result_gain_df_arr,year_labels):
            model_score_s['total_get_year{year}'.format(year=label)]=year_df["gain"].sum()
            model_score_s['total_use_year{year}'.format(year=label)]=100*year_df["pred"].sum()
            model_score_s['num_com_year{year}'.format(year=label)]=year_df['trans_result'].sum()
            model_score_s['num_pred_year{year}'.format(year=label)]=year_df['pred'].sum()
            model_score_s['gain_year{year}'.format(year=label)]=(model_score_s['total_get_year{year}'.format(year=label)]/model_score_s['total_use_year{year}'.format(year=label)])*100
            model_score_s['num_hit_year{year}'.format(year=label)]=year_df['hit'].sum()
            model_score_s['buy_hit_per_year{year}'.format(year=label)]=(model_score_s['num_hit_year{year}'.format(year=label)]/ model_score_s['num_pred_year{year}'.format(year=label)])*100
        model_score_df=model_score_df.append(model_score_s,ignore_index=True,sort=False)
    #モデルの「スコアを保存
    dir_path =  "../../bot_database/{place_name}/model_score_{place_name}/check_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None#model_selectionで決定したぱらめーたをもとにモデルを保存。
