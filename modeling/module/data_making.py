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

#自作のモジュールのインポート
import module.master as master
import module.graph as graph
import module.trans_text_code as trans
#import data_making as making
import module.model_analysis as analysis


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

#閾値を渡して、その値以上を1、未満を0に置き変える。
def pred_th_trans(pred_df,th):
    #引数として予測結果のdeと、変換したい閾値を渡す。
    trans_df=pred_df.copy()
    trans_df.loc[trans_df['pred_proba'] >= th, 'pred'] = 1
    trans_df.loc[~(trans_df['pred_proba']  >=  th), 'pred'] = 0
    return trans_df

#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#基本的に分析用になるのかな？大会の開催日数や四半期、大会の中の何日目かの情報をもったdfを返す。
def get_event_info(result_base_df):
    df=result_base_df.copy()
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
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#モデル作成関数
#モデル作成関数
#モデル作成関数
#モデル作成関数
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================



#モデルのパラメータ探索関数(XGboost)
def model_score_XGboost(version,place_name,result_df):#学習データと場所名を渡せば探索を初めて、指定のディレクトリにスコアをまとめたcsvを出力する。
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf

    #学習データの切り分け
    test_df = result_df[(result_df['year']==2019) | ((result_df['year']==2020) )]#2019,2020のデータを検証用データに。
    train_df =  result_df[(result_df['year']!=2019) & ((result_df['year']!=2020) )]#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    test_df=test_df.drop(['year'],axis=1)
    train_df=train_df.drop(['year'],axis=1)

    train_money=pd.Series(train_df['money'])
    test_money=pd.Series(test_df['money'])

    #x,yへの切り分け
    #出現数の分布
    result_com_s=test_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    gain_mean=test_df.groupby('result_com')['money'].mean()
    gain_mean=gain_mean.sort_index()

    gain_median=test_df.groupby('result_com')['money'].median()
    gain_median=gain_median.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index,
                                'result_com_num':result_com_s.values,
                                'result_com_per':result_com_s.values/sum(result_com_s.values)*100,
                                'gain_mean':gain_mean.values,
                                'gain_median':gain_median.values,})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため




    for result_com_number in tqdm(result_com_df['result_com'].values):
        #print(result_com_number)
        result_com=result_com_number
        #result_comごとの閾値の決定========================================================================
        #print(result_com_number)
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================

        gain_th=10#利益率の閾値
        result_s=result_com_df[result_com_df['result_com']==result_com]
        buy_accuracy_th=result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値
        num_tp_th=result_s['result_com_num'].values[0]*0.2#あたった回数の閾値(出現回数の20%が的中)
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

        for_arr=np.arange(1,73)
        #for_arr=np.arange(1,100,1)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        model_gain_arr=[0]*len(result_test_df)
        test_gain_arr=test_money.values
        #depths_arr=[4,5,6,7,8]
        depths_arr=[5,6,8]
        for depth in depths_arr:
            for sum_target_per in for_arr:

                index=sum_target_per-1
                #target_per=50+sum_target_per
                target_per=80+(sum_target_per*3)
                target_per_arr[index]=target_per

                #モデルの評価指標値を格納するseries======================
                model_score_s=pd.Series(index=['target_com','depth','target_per','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df#ベースのデータフレームをコピー
                target_df=target_df.sample(frac=1, random_state=1)#シャッフル、時系列の偏りを無くす
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
                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=True, random_state=7)#学習データ内でさらに分割してロスをもとに修正をする。

                #XGboostのデータ型に変換する
                train = xgb.DMatrix(train_x, label=train_y)#学習用
                valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                test = xgb.DMatrix(target_x_test, label=target_y_test)#実際に使った時の利益率の算出用

                #xgb.config_context(verbosity=0)
                param = {'max_depth': depth, #パラメータの設定
                         'eta': 0.5,
                         #'eta': 0.2,
                         'objective': 'binary:hinge',
                         'eval_metric': 'logloss',
                         'verbosity':0,
                         'subsample':0.8,
                         'nthread':10,
                         'gpu_id':0,
                         'tree_method':'gpu_hist'
                        }
                evallist = [(valid, 'eval'), (train, 'train')]#学習時にバリデーションを監視するデータの指定。
                #bst = xgb.train(param, train,num_boost_round=1000,early_stopping_rounds=30)
                num_round = 10000
                bst = xgb.train(param, train,num_round,evallist, early_stopping_rounds=30, verbose_eval=0 )
                #RF = RandomForestClassifier(random_state=1,n_estimators=1000,max_depth=depth)
                #RF = RF.fit(target_x_train,target_y_train)


                # 未知データに対する予測値
                #predict_y_test = RF.predict(target_x_test)
                predict_y_test=bst.predict(test)

                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================

                #[1]の正答率を見る
                pred_test_df=pd.DataFrame({'pred':predict_y_test
                                          , 'test':target_y_test})
                num_1=len(pred_test_df[pred_test_df['test']==1])
                count=0
                #追加　配当金の情報も考慮する。
                gain_index=0
                model_gain_arr=[0]*len(result_test_df)
                for _, s in pred_test_df.iterrows():
                    if ((s['pred']==1) and (s['test']==1)):
                        count+=1#的中回数
                        model_gain_arr[gain_index]=test_gain_arr[gain_index]
                    gain_index+=1
                #print('test accyracy: {}'.format((count/num_1)*100))
                gain_arr[index]=sum(model_gain_arr)
                accuracy_arr[index]=(count/num_1)*100
                try:
                    pred_0[index]=pred_test_df['pred'].value_counts()[0]
                except:
                    pred_0[index]=0
                #scoreのseriesに情報書き込み==================
                model_score_s['総収益']=sum(model_gain_arr)
                model_score_s['投資金額']=100*sum(predict_y_test)
                model_score_s['出現数']=sum(target_y_test)
                model_score_s['購買予測数']=sum(predict_y_test)
                model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                model_score_s['購買的中率']=(count/sum(predict_y_test))*100
                model_score_s['的中数']=count
                model_score_df=model_score_df.append(model_score_s,ignore_index=True)


    #モデルの「スコアを保存
    #model_score_df.to_csv('{}_model_score.csv'.format(place), encoding='utf_8_sig')
    dir_path = "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None







#モデルのパラメータ探索関数(XGboost)
def model_score_XGboost_th(version,place_name,result_df):#XGboostの出力を確率のやつを使用したバージョン、閾値の探索も行う。
    print(place_name)
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

    #x,yへの切り分け
    #出現数の分布
    result_com_s=test_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    gain_mean=test_df.groupby('result_com')['money'].mean()
    gain_mean=gain_mean.sort_index()

    gain_median=test_df.groupby('result_com')['money'].median()
    gain_median=gain_median.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index,
                                'result_com_num':result_com_s.values,
                                'result_com_per':result_com_s.values/sum(result_com_s.values)*100,
                                'gain_mean':gain_mean.values,
                                'gain_median':gain_median.values,})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため




    for result_com_number in tqdm(result_com_df['result_com'].values):
        #print(result_com_number)
        result_com=result_com_number
        #result_comごとの閾値の決定========================================================================
        #print(result_com_number)
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================

        gain_th=10#利益率の閾値
        result_s=result_com_df[result_com_df['result_com']==result_com]
        buy_accuracy_th=result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値
        num_tp_th=result_s['result_com_num'].values[0]*0.2#あたった回数の閾値(出現回数の20%が的中)
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
        #depths_arr=[4,5,6,7,8]
        #depths_arr=[5,6,8]
        depths_arr=[5,8]
        for depth in depths_arr:
            for sum_target_per in for_arr:

                index=sum_target_per-1
                #target_per=50+sum_target_per
                target_per=100+(sum_target_per*2)
                target_per_arr[index]=target_per

                #モデルの評価指標値を格納するseries======================
                model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df#ベースのデータフレームをコピー
                target_df=target_df.sample(frac=1, random_state=7)#シャッフル、時系列の偏りを無くす
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
                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=True, random_state=7)#学習データ内でさらに分割してロスをもとに修正をする。

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
                #bst = xgb.train(param, train,num_boost_round=1000,early_stopping_rounds=30)
                #num_round = 10000
                num_round = 400
                bst = xgb.train(param, train,num_round,evallist, early_stopping_rounds=30, verbose_eval=0 )
                #bst = xgb.train(param, train,num_round,evallist, verbose=100,early_stopping_rounds=30 )
                #RF = RandomForestClassifier(random_state=1,n_estimators=1000,max_depth=depth)
                #RF = RF.fit(target_x_train,target_y_train)


                # 未知データに対する予測値
                #predict_y_test = RF.predict(target_x_test)
                predict_y_test=bst.predict(test)

                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================

                #[1]の正答率を見る
                pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                                          , 'test':target_y_test})

                #th_arr=[0.1,0.3,0.5,0.6,0.7,0.8,0.9]
                #th_arr=[0.01,0.03,0.05,0.07,0.9,0.1,0.13]#探索結果待ち、、、、、
                th_arr=[0.85,0.9,0.92]
                for th in th_arr:
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
    dir_path = "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None




def V2_1_check(version,place_name,result_df):#XGboostおかしい部分を探す
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf

    #学習データの切り分け
    #test_df = result_df[(result_df['year']==2019) | ((result_df['year']==2020) )]#2019,2020のデータを検証用データに。
    train_df =  result_df[(result_df['year']!=2019) & ((result_df['year']!=2020) )]#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    #test_df=test_df.drop(['year'],axis=1)
    display(train_df.head())
    display(train_df.tail())
    train_df=train_df.drop(['year'],axis=1)

    train_money=pd.Series(train_df['money'])
    #test_money=pd.Series(test_df['money'])

    #出現数の分布
    #result_com_s=test_df['result_com'].value_counts()
    result_com_s=train_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    #gain_mean=test_df.groupby('result_com')['money'].mean()
    gain_mean=train_df.groupby('result_com')['money'].mean()
    gain_mean=gain_mean.sort_index()

    #gain_median=test_df.groupby('result_com')['money'].median()
    gain_median=train_df.groupby('result_com')['money'].median()
    gain_median=gain_median.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index,
                                'result_com_num':result_com_s.values,
                                'result_com_per':result_com_s.values/sum(result_com_s.values)*100,
                                'gain_mean':gain_mean.values,
                                'gain_median':gain_median.values,})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため




    for result_com_number in tqdm(result_com_df['result_com'].values):
        #print(result_com_number)
        result_com=result_com_number
        #result_comごとの閾値の決定========================================================================
        #print(result_com_number)
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================

        gain_th=10#利益率の閾値
        result_s=result_com_df[result_com_df['result_com']==result_com]
        buy_accuracy_th=result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値
        num_tp_th=result_s['result_com_num'].values[0]*0.2#あたった回数の閾値(出現回数の20%が的中)
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


        #result_test_df=test_df.copy()
        #result_arr=[0]*len(result_test_df)
        # i=0
        # for result in result_test_df['result_com']:
        #     if ((result==result_com)):
        #         result_arr[i]=1
        #     else:
        #         result_arr[i]=0
        #     i+=1

        #result_test_df['result_com']=result_arr

        #result_train_df['money']=train_money
        #result_test_df['money']=test_money
        #学習データラベル変換終わり============================================

        for_arr=np.arange(1,85)
        #for_arr=np.arange(1,100,1)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        #model_gain_arr=[0]*len(result_test_df)
        #test_gain_arr=test_money.values
        #depths_arr=[4,5,6,7,8]
        #depths_arr=[5,6,8]
        depths_arr=[5,8]
        for depth in depths_arr:
            for sum_target_per in for_arr:

                index=sum_target_per-1
                #target_per=50+sum_target_per
                target_per=100+(sum_target_per*2)
                target_per_arr[index]=target_per

                #モデルの評価指標値を格納するseries======================
                model_score_s=pd.Series(index=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df#ベースのデータフレームをコピー
                target_df=target_df.sample(frac=1, random_state=7)#シャッフル、時系列の偏りを無くす
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
                #target_x_test=result_test_df.drop('money',axis=1)
                #target_x_test=target_x_test.drop('result_com',axis=1)

                target_y_train=target_train_df['result_com']
                #target_y_test=result_test_df['result_com']
                train_x, valid_x, train_y, valid_y = train_test_split(target_x_train, target_y_train, test_size=0.2, shuffle=True, random_state=7)#学習データ内でさらに分割してロスをもとに修正をする。
                print('len_train',len(train_y))
                print('len_valid',len(valid_y))
                #XGboostのデータ型に変換する
                train = xgb.DMatrix(train_x, label=train_y)#学習用
                valid = xgb.DMatrix(valid_x, label=valid_y)#学習時のロス修正用
                #test = xgb.DMatrix(target_x_test, label=target_y_test)#実際に使った時の利益率の算出用

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
                                 'nthread':5,
                                 'gpu_id':0,
                                 'seed':7,
                                 'tree_method':'gpu_hist'
                                }
                evallist = [(valid, 'eval'), (train, 'train')]#学習時にバリデーションを監視するデータの指定。
                #bst = xgb.train(param, train,num_boost_round=1000,early_stopping_rounds=30)
                #num_round = 10000
                num_round = 400
                bst = xgb.train(param, train,num_round,evallist, early_stopping_rounds=30, verbose_eval=0 )
                #bst = xgb.train(param, train,num_round,evallist, verbose=100,early_stopping_rounds=30 )
                #RF = RandomForestClassifier(random_state=1,n_estimators=1000,max_depth=depth)
                #RF = RF.fit(target_x_train,target_y_train)


                # 未知データに対する予測値
                #predict_y_test = RF.predict(target_x_test)
                #predict_y_test=bst.predict(test)

                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================

                #[1]の正答率を見る
                #pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                #                          , 'test':target_y_test})

                #th_arr=[0.1,0.3,0.5,0.6,0.7,0.8,0.9]
                #th_arr=[0.01,0.03,0.05,0.07,0.9,0.1,0.13]#探索結果待ち、、、、、
    #             th_arr=[0.85,0.9,0.92]
    #             for th in th_arr:
    #                 trans_df=pred_th_trans(pred_test_df,th)
    #                 num_1=len(trans_df[trans_df['test']==1])
    #                 count=0
    #                 #追加　配当金の情報も考慮する。
    #                 gain_index=0
    #                 model_gain_arr=[0]*len(result_test_df)
    #                 for _, s in trans_df.iterrows():
    #                     if ((s['pred']==1) and (s['test']==1)):#もし購買しているかつ的中をしていたら・・・
    #                         count+=1#的中回数
    #                         model_gain_arr[gain_index]=test_gain_arr[gain_index]
    #                     gain_index+=1
    #                 #print('test accyracy: {}'.format((count/num_1)*100))
    #                 gain_arr[index]=sum(model_gain_arr)
    #                 accuracy_arr[index]=(count/num_1)*100
    #                 try:
    #                     pred_0[index]=trans_df['pred'].value_counts()[0]
    #                 except:
    #                     pred_0[index]=0
    #                 #scoreのseriesに情報書き込み==================
    #                 model_score_s['threshold']=th
    #                 model_score_s['総収益']=sum(model_gain_arr)
    #                 #model_score_s['投資金額']=100*sum(predict_y_test)
    #                 model_score_s['投資金額']=100*trans_df['pred'].sum()
    #                 model_score_s['出現数']=sum(target_y_test)
    #                 #model_score_s['購買予測数']=sum(predict_y_test)
    #                 model_score_s['購買予測数']=trans_df['pred'].sum()
    #                 model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
    #                 model_score_s['購買的中率']=(count/trans_df['pred'].sum())*100
    #                 model_score_s['的中数']=count
    #                 model_score_df=model_score_df.append(model_score_s,ignore_index=True)
    # #モデルの「スコアを保存
    # #model_score_df.to_csv('{}_model_score.csv'.format(place), encoding='utf_8_sig')
    # dir_path = "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    # model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None











def model_score_rondom_forest(version,place_name,result_df):#学習データと場所名を渡せば探索を初めて、指定のディレクトリにスコアをまとめたcsvを出力する。
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf

    #学習データの切り分け
    test_df = result_df[(result_df['year']==2019) | ((result_df['year']==2020) )]#2019,2020のデータを検証用データに。
    train_df =  result_df[(result_df['year']!=2019) & ((result_df['year']!=2020) )]#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    test_df=test_df.drop(['year'],axis=1)
    train_df=train_df.drop(['year'],axis=1)

    train_money=pd.Series(train_df['money'])
    test_money=pd.Series(test_df['money'])

    #x,yへの切り分け
    #出現数の分布
    result_com_s=test_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    gain_mean=test_df.groupby('result_com')['money'].mean()
    gain_mean=gain_mean.sort_index()

    gain_median=test_df.groupby('result_com')['money'].median()
    gain_median=gain_median.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index,
                                'result_com_num':result_com_s.values,
                                'result_com_per':result_com_s.values/sum(result_com_s.values)*100,
                                'gain_mean':gain_mean.values,
                                'gain_median':gain_median.values,})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため




    for result_com_number in tqdm(result_com_df['result_com'].values):
        #print(result_com_number)
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================

        gain_th=10#利益率の閾値
        result_s=result_com_df[result_com_df['result_com']==result_com]
        buy_accuracy_th=result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値
        num_tp_th=result_s['result_com_num'].values[0]*0.2#あたった回数の閾値(出現回数の20%が的中)
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

        for_arr=np.arange(1,73)
        #for_arr=np.arange(1,100,1)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        model_gain_arr=[0]*len(result_test_df)
        test_gain_arr=test_money.values
        #depths_arr=[4,5,6,7,8]
        depths_arr=[5,6,8]
        for depth in depths_arr:
            for sum_target_per in for_arr:

                index=sum_target_per-1
                #target_per=50+sum_target_per
                target_per=80+(sum_target_per)
                target_per_arr[index]=target_per

                #モデルの評価指標値を格納するseries======================
                model_score_s=pd.Series(index=['target_com','depth','target_per','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df#ベースのデータフレームをコピー
                target_df=target_df.sample(frac=1, random_state=1)#シャッフル、時系列の偏りを無くす
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

                #テストデータ
                RF = RandomForestClassifier(random_state=1,n_estimators=1000,max_depth=depth,n_jobs=10)
                RF = RF.fit(target_x_train,target_y_train)
                # 未知データに対する予測値
                predict_y_test = RF.predict(target_x_test)
                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================

                #[1]の正答率を見る
                pred_test_df=pd.DataFrame({'pred':predict_y_test
                                          , 'test':target_y_test})
                num_1=len(pred_test_df[pred_test_df['test']==1])
                count=0
                #追加　配当金の情報も考慮する。
                gain_index=0
                model_gain_arr=[0]*len(result_test_df)
                for _, s in pred_test_df.iterrows():
                    if ((s['pred']==1) and (s['test']==1)):
                        count+=1#的中回数
                        model_gain_arr[gain_index]=test_gain_arr[gain_index]
                    gain_index+=1
                #print('test accyracy: {}'.format((count/num_1)*100))
                gain_arr[index]=sum(model_gain_arr)
                accuracy_arr[index]=(count/num_1)*100
                try:
                    pred_0[index]=pred_test_df['pred'].value_counts()[0]
                except:
                    pred_0[index]=0
                #scoreのseriesに情報書き込み==================
                model_score_s['総収益']=sum(model_gain_arr)
                model_score_s['投資金額']=100*sum(predict_y_test)
                model_score_s['出現数']=sum(target_y_test)
                model_score_s['購買予測数']=sum(predict_y_test)
                model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
                model_score_s['購買的中率']=(count/sum(predict_y_test))*100
                model_score_s['的中数']=count
                model_score_df=model_score_df.append(model_score_s,ignore_index=True)


    #モデルの「スコアを保存
    #model_score_df.to_csv('{}_model_score.csv'.format(place), encoding='utf_8_sig')
    dir_path = "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None



def model_score_rondom_forest_th(version,place_name,result_df):#学習データと場所名を渡せば探索を初めて、指定のディレクトリにスコアをまとめたcsvを出力する。(rondomorestのperdict_proba版)
    print(place_name)
    #result_dfは加工関数にて分けられたものを渡す。
    model_score_df=pd.DataFrame(columns=['target_com','depth','target_per','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])#スコアを格納するdf

    #学習データの切り分け
    test_df = result_df[(result_df['year']==2019) | ((result_df['year']==2020) )]#2019,2020のデータを検証用データに。
    train_df =  result_df[(result_df['year']!=2019) & ((result_df['year']!=2020) )]#そのほかを学習データに
    #学習データを切り分けたらyearはいらないから削除する
    test_df=test_df.drop(['year'],axis=1)
    train_df=train_df.drop(['year'],axis=1)

    train_money=pd.Series(train_df['money'])
    test_money=pd.Series(test_df['money'])

    #x,yへの切り分け
    #出現数の分布
    result_com_s=test_df['result_com'].value_counts()
    result_com_s=result_com_s.sort_index()
    gain_mean=test_df.groupby('result_com')['money'].mean()
    gain_mean=gain_mean.sort_index()

    gain_median=test_df.groupby('result_com')['money'].median()
    gain_median=gain_median.sort_index()
    result_com_df=pd.DataFrame({'result_com':result_com_s.index,
                                'result_com_num':result_com_s.values,
                                'result_com_per':result_com_s.values/sum(result_com_s.values)*100,
                                'gain_mean':gain_mean.values,
                                'gain_median':gain_median.values,})
    result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため




    for result_com_number in tqdm(result_com_df['result_com'].values):
        #print(result_com_number)
        result_com=result_com_number

        #result_comごとの閾値の決定========================================================================

        gain_th=10#利益率の閾値
        result_s=result_com_df[result_com_df['result_com']==result_com]
        buy_accuracy_th=result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値
        num_tp_th=result_s['result_com_num'].values[0]*0.2#あたった回数の閾値(出現回数の20%が的中)
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

        for_arr=np.arange(1,73)
        #for_arr=np.arange(1,100,1)
        accuracy_arr=[0]*len(for_arr)
        target_per_arr=[0]*len(for_arr)
        pred_0=[0]*len(for_arr)
        gain_arr=[0]*len(for_arr)
        model_gain_arr=[0]*len(result_test_df)
        test_gain_arr=test_money.values
        #depths_arr=[4,5,6,7,8]
        depths_arr=[5,7]
        for depth in depths_arr:
            for sum_target_per in for_arr:

                index=sum_target_per-1
                #target_per=50+sum_target_per
                target_per=80+(sum_target_per)
                target_per_arr[index]=target_per

                #モデルの評価指標値を格納するseries======================
                model_score_s=pd.Series(index=['target_com','depth','target_per','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'], dtype='float64')
                model_score_s['target_com']=result_com#目標としているresult_comラベル番号
                model_score_s['depth']=depth#ハイパーパラメータ＿木の深さ
                model_score_s['target_per']=target_per#学習データ_1に対してどの程度の0のデータを持たせるか。
                #======================
                #trainの[0]に対して、target_perの割合の量[1]を持った学習データの作成
                # 一層目の判別機のtrainデータ　:terget_result_df
                target_df=result_train_df#ベースのデータフレームをコピー
                target_df=target_df.sample(frac=1, random_state=1)#シャッフル、時系列の偏りを無くす
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

                #テストデータ
                RF = RandomForestClassifier(random_state=1,n_estimators=1000,max_depth=depth,n_jobs=5)
                RF = RF.fit(target_x_train,target_y_train)


                # 未知データに対する予測値(確率で出力)
                predict_y_test_proba_arr = RF.predict_proba(target_x_test)#まだ多次元リスト
                predict_y_test_check=RF.predict(target_x_test)#チェック用、ラベルの順番を確認
                predict_y_test=[proba_arr[1] for proba_arr in predict_y_test_proba_arr]#1にあたる部分の確率のみ出力
                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================
                #==========================================================================================================================================

                #[1]の正答率を見る
                pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                                          , 'test':target_y_test})

                #th_arr=[0.5,0.55,0.6,0.65,0.7,0.8,0.9]
                th_arr=[0.5,0.52,0.54,0.56,0.58]
                for th in th_arr:
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
    dir_path = "../../bot_database/{place_name}/model_score_{place_name}/{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_score_df=model_score_df.sort_values(['target_com', 'depth','threshold','target_per'])#並べ替え
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None


#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================


#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#学習データ作成関数
#学習データ作成関数
#学習データ作成関数
#学習データ作成関数
#学習データ作成関数
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================
#==================================================================================================================================================================

#train_{}のscvを突っ込むと以下の加工をする
#dateをけして、yearを追加
#各変数のダミー化
def data_making_mo_bo(df):#クラスタリングなし、ボート、艇番号あり

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




#バージョン1_1、()データ切り抜き関数配当金、着の情報は切りぬかなくてもうまいことやってくれる。===============================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#train_{}のscvを突っ込むと以下の加工をする
#dateをけして、yearを追加
#各変数のダミー化
#学習データからクラスタリングラベル、次元削減の付与,
#また、ボート番号、モータ番号を消す。

def data_making_clustar(df):#クラスタリングあり、モータ番号、艇番号なし


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
        Km = KMeans(random_state=7,n_clusters=num_cluster).fit(clustar_train_df)#rondom_stateはラッキーセブン
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
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#モデルのセーブ部分
#モデルのセーブ部分
#モデルのセーブ部分
#モデルのセーブ部分


def save_model_V2_1(result_base_df,use_model_df,place_name,version):
    #探査結果から学習したモデルを保存する関数、
    print(place_name)
    #==============================================================================
    #学習関数で場所ごとにバージョンに対応した学習データを作る
    result_df=data_making_clustar(result_base_df)
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

    # #x,yへの切り分け
    # #出現数の分布
    # result_com_s=test_df['result_com'].value_counts()
    # result_com_s=result_com_s.sort_index()
    # gain_mean=test_df.groupby('result_com')['money'].mean()
    # gain_mean=gain_mean.sort_index()
    #
    # gain_median=test_df.groupby('result_com')['money'].median()
    # gain_median=gain_median.sort_index()
    # result_com_df=pd.DataFrame({'result_com':result_com_s.index,
    #                             'result_com_num':result_com_s.values,
    #                             'result_com_per':result_com_s.values/sum(result_com_s.values)*100,
    #                             'gain_mean':gain_mean.values,
    #                             'gain_median':gain_median.values,})
    # result_com_df=result_com_df.iloc[0:28]#探索的に探すにも最後のほうは役にモデルなのはわかっているため
    for index, model_row in use_model_df.iterrows():
        #パラメータ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        #======================================================================================
        #result_com=int(model_row['target_com'])
        result_com=int(model_row['target_com'])
        depth=int(model_row['depth'])
        target_per=int(model_row['target_per'])
        th=float(model_row['threshold'])

        #======================================================================================
        #======================================================================================
        #======================================================================================


        # gain_th=10#利益率の閾値
        # result_s=result_com_df[result_com_df['result_com']==result_com]
        # buy_accuracy_th=result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値
        # num_tp_th=result_s['result_com_num'].values[0]*0.2#あたった回数の閾値(出現回数の20%が的中)
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
        bst = xgb.train(param, train,num_round,evallist, early_stopping_rounds=30, verbose_eval=0 )

        # 未知データに対する予測値
        #predict_y_test = RF.predict(target_x_test)
        predict_y_test=bst.predict(test)


        #==========================================================================================================================================
        #==========================================================================================================================================

        #pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/com{com}_{depth}_{target_per}_{th}_{place_name}.pickle".format(place_name=place_name,com=result_com,depth=depth,target_per=target_per,th=th)#モデルを保存
        pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/com{com}_{depth}_{target_per}_{th}_{place_name}.sav".format(place_name=place_name,com=result_com,depth=depth,target_per=target_per,th=th)#モデルを保存
        #print(pickle_path)
        pickle.dump(bst, open(pickle_path, "wb"))#モデルの保存
        #with open(pickle_path, 'wb') as model_file:
        #    pickle.dump(bst, model_file)
        #==========================================================================================================================================
        #==========================================================================================================================================



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
    dir_path = "../bot_database/{place_name}/model_score_{place_name}/check_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')
    return None








#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#====================================================================================================================================================================================================
#バージョン管理部分======================================================================================
#バージョン管理部分======================================================================================
#バージョン管理部分======================================================================================
#バージョン管理部分======================================================================================
def version_2_0(version,place_name,base_df):
    result_df=data_making_clustar(base_df)
    #display(result_df)
    model_score_rondom_forest(version,place_name,result_df)


def version_2_1(version,place_name,base_df):#閾値で予測を変えるバージョン
    result_df=data_making_clustar(base_df)
    model_score_XGboost_th(version,place_name,result_df)#閾値を決めて変換するver


def version_2_1_1(version,place_name,base_df):#閾値で予測を変えるバージョンのrandom_forest版
    result_df=data_making_clustar(base_df)
    V2_1_check(version,place_name,result_df)#閾値を決めて変換するver


def version_2_2(version,place_name,base_df):#閾値で予測を変えるバージョンのrandom_forest版
    result_df=data_making_clustar(base_df)
    model_score_rondom_forest_th(version,place_name,result_df)#閾値を決めて変換するver
