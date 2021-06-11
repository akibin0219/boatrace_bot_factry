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


#モデルの動作確認を行う際に使う関数=========================================================================================================================
#モデルの動作確認を行う際に使う関数=========================================================================================================================
#モデルの動作確認を行う際に使う関数=========================================================================================================================
#モデルの動作確認を行う際に使う関数=========================================================================================================================
#モデルの動作確認を行う際に使う関数=========================================================================================================================
#モデルの動作確認を行う際に使う関数=========================================================================================================================

def data_making_clustar_pickle(df,place_name,version):#クラスタリングあり、モータ番号、艇番号なし,oickleのを読み込む
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
        #Km = KMeans(random_state=7,n_clusters=num_cluster).fit(clustar_train_df)#rondom_stateはラッキーセブン
        pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/clustering_{place_name}_num_{num_cluster}_{V}.sav".format(place_name=place_name,num_cluster=num_cluster,V=version)#モデルを保存
        Km =pickle.load(open(pickle_path, 'rb'))
        train_pred = Km.predict(clustar_train_df)#rondom_stateはラッキーセブン
        test_pred =Km.predict(clustar_test_df)#rondom_stateはラッキーセブン
        #Km=========================実査に使うときはこれのモデルを会場ごとに保存して使用。
        clustar_train_df['num={}'.format(num_cluster)]=train_pred
        clustar_test_df['num={}'.format(num_cluster)]=test_pred

    #結合して元の形に戻す。
    clustar_df=pd.concat([clustar_train_df, clustar_test_df])
    clustar_df['year']=years
    clustar_df['money']=money
    clustar_df['result_com']=result

    model_df=clustar_df
    return model_df



def check_model_V2_1(result_base_df,use_model_df,place_name,version):#モデルをすべてpickleで読み込んで精度検証、
    print(place_name)
    #==============================================================================
    #学習関数で場所ごとにバージョンに対応した学習データを作る
    result_df=data_making_clustar_pickle(result_base_df,place_name,version)
    #==============================================================================
    #result_dfは加工関数にて分けられたものを渡す。
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf

    #学習データの切り分け
    test_df = result_df[(result_df['year']==2019) | ((result_df['year']==2020) )]#2019,2020のデータを検証用データに。
    train_df =  result_df[(result_df['year']!=2019) & ((result_df['year']!=2020) )]#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    test_df=test_df.drop(['year'],axis=1)
    train_df=train_df.drop(['year'],axis=1)

    train_money=pd.Series(train_df['money'])
    test_money=pd.Series(test_df['money'])
    for index, model_row in use_model_df.iterrows():
        #パラメータ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        #======================================================================================
        #result_com=int(model_row['target_com'])
        result_com=int(model_row['target_com'])
        depth=int(model_row['depth'])
        target_per=int(model_row['target_per'])
        th=float(model_row['threshold'])

        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_arr=[0]*len(result_train_df)
        i=0
        for result in result_train_df['result_com']:
            if ((result==result_com)):
                result_arr[i]=1
            else:
                result_arr[i]=0
            i+=1
        result_train_df['result_com']=result_arr
        result_test_df=test_df.copy()
        result_arr=[0]*len(result_test_df)
        i=0
        for result in result_test_df['result_com']:
            if ((result==result_com)):
                result_arr[i]=1
            else:
                result_arr[i]=0
            i+=1

        result_test_df['result_com']=result_arr

        result_train_df['money']=train_money
        result_test_df['money']=test_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,85)
        #for_arr=np.arange(1,100,1)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        model_gain_arr=[0]*len(result_test_df)
        test_gain_arr=test_money.values


        #モデルの評価指標値を格納するseries======================
        model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
        model_score_s['target_com']=result_com#目標としているresult_comラベル番号
        model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
        model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
        #======================
        #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
        # 一層目の判別機のtrainデータ　:terget_result_df
        target_df=result_train_df#ベースのデータフレームをコピー
        target_df=target_df.sample(frac=1,random_state=7)#シャッフル、時系列の偏りを無くす
        target_1_df=target_df[target_df['result_com']==1]
        len_1=len(target_1_df)
        target_0_df=target_df[target_df['result_com']==0]
        len_0=len(target_0_df)
        target_0_df=target_0_df.iloc[(len_0-int(len_1*(target_per/100))):len_0]#1に対する目標の割合ぶん0の結果だったレースを抽出（後ろから抽出）
        target_train_df=pd.concat([target_1_df, target_0_df])
        #学習＆予測ぱーと========================================================================
        #==========================================================================================================================================
        #データの切り分け
        target_x_train=target_train_df.drop('money',axis=1)
        target_x_train=target_x_train.drop('result_com',axis=1)
        target_x_test=result_test_df.drop('money',axis=1)
        target_x_test=target_x_test.drop('result_com',axis=1)

        target_y_train=target_train_df['result_com']
        target_y_test=result_test_df['result_com']
        train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2,shuffle=True, random_state=7)#学習データ内でさらに分割してロスをもとに修正をする。

        #XGboostのデータ型に変換する
        train = xgb.DMatrix(train_x, label=train_y)#学習用
        valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
        test = xgb.DMatrix(target_x_test, label=target_y_test)#実際に使った時の利益率の算出用

        #xgb.config_context(verbosity=0)
        param = {'max_depth': depth, #パラメータの設定
                         #'eta': 1.8,
                         #'eta': 0.8,
                         'eta': 1.3,
                         #'eta': 0.2,
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
        evallist = [(valid, 'eval'), (train, 'train')]#学習時にバリデーションを監視するデータの指定。
        num_round = 400
        pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/com{com}_{depth}_{target_per}_{th}_{place_name}.sav".format(place_name=place_name,com=result_com,depth=depth,target_per=target_per,th=th)#モデルを保存

        bst = pickle.load(open(pickle_path, 'rb'))
        # 未知データに対する予測値
        #predict_y_test = RF.predict(target_x_test)
        predict_y_test=bst.predict(test)
        #[1]の正答率を見る
        pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                                  , 'test':target_y_test})
        trans_df=pred_th_trans(pred_test_df,th)
        num_1=len(trans_df[trans_df['test']==1])
        count=0
        #追加　配当金の情報も考慮する。
        gain_index=0
        model_gain_arr=[0]*len(result_test_df)
        for _, s in trans_df.iterrows():
            if ((s['pred']==1) and (s['test']==1)):#もし購買しているかつ的中をしていたら・・・
                count+=1#的中回数
                model_gain_arr[gain_index]=test_gain_arr[gain_index]
            gain_index+=1
        #print('test accyracy: {}'.format((count/num_1)*100))
        gain_arr[index]=sum(model_gain_arr)
        accuracy_arr[index]=(count/num_1)*100
        try:
            pred_0[index]=trans_df['pred'].value_counts()[0]
        except:
            pred_0[index]=0
        #scoreのseriesに情報書き込み==================
        model_score_s['threshold']=th
        model_score_s['総収益']=sum(model_gain_arr)
        #model_score_s['投資金額']=100*sum(predict_y_test)
        model_score_s['投資金額']=100*trans_df['pred'].sum()
        model_score_s['出現数']=sum(target_y_test)
        #model_score_s['購買予測数']=sum(predict_y_test)
        model_score_s['購買予測数']=trans_df['pred'].sum()
        model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
        model_score_s['購買的中率']=(count/trans_df['pred'].sum())*100
        model_score_s['的中数']=count
        model_score_df=model_score_df.append(model_score_s,ignore_index=True)
    #モデルの「スコアを保存
    #model_score_df.to_csv('{}_model_score.csv'.format(place), encoding='utf_8_sig')
    dir_path = "../bot_database/{place_name}/model_score_{place_name}/check_{place_name}_all_pickle_{V}.csv".format(place_name=place_name,V=version)#すべてpickleバージョンの読み込み
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None
