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
import module.data_making as making
#import model_analysis as analysis


from sklearn.preprocessing import StandardScaler#モデルの評価用に標準化する関数
import scipy.stats#モデルの評価用に標準化する関数

pd.set_option('display.width',400)#勝手に改行コードを入れられるのを防ぐ

def model_analysis(score_df,place_name,version):
    #基準をクリアしたモデルを格納するdf
    depths_arr=[5,8]
    th_arr=[0.85,0.9,0.92]
    #depths_arr=[4,5,6,7,8]
    result_com_arr=np.arange(1, 29)

    #基本敵に今回からは出現数とかはresult_comのdfから作成しないでスコアのdfからしゅとくする。

    model_para_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])
    #model_para_df=pd.DataFrame(columns=['target_com','depth','target_per','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数'])
    for target_com in result_com_arr:
        target_com_df=score_df[score_df['target_com']==target_com]
        num_com= target_com_df['出現数'].values[0]  #出現数
        #閾値の作成===============================================================
        #gain_th=110#利益率の閾値(メインモデル)
        gain_th=115#利益率の閾値(メインモデル)
        #result_s=result_com_df[result_com_df['result_com']==target_com]
        buy_accuracy_th=5#result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値(いったん決めで5%とする)
        num_tp_th=num_com*0.15#あたった回数の閾値(出現回数の15%が的中)

        diff_gain_th=105#利益率の閾値(前後モデル)
        diff_buy_accuracy_th=5#result_s['result_com_per'].values[0]*1.1#買ったうちの的中率の閾値(いったん決めで5%とする)
        diff_num_tp_th=num_com*0.15#あたった回数の閾値(出現回数の15%が的中)

        #==========================================================
        for target_depth in depths_arr:
            target_com_depth_df=target_com_df[target_com_df['depth']==target_depth]

            for th in th_arr:
                target_com_depth_th_df=target_com_depth_df[target_com_depth_df['threshold']==th]

                len_df=len(target_com_depth_th_df)
                df=target_com_depth_th_df.copy()
                df['number_i']=np.arange(0,len_df,1)
                #display(df)
                for _, row in df.iterrows():
                    #if ((row['number_i']==0) or (row['number_i']==1) or (row['number_i']==len_df-2) or (row['number_i']==len_df-1)):
                    if ((row['number_i']==0) or (row['number_i']==len_df-1)):#前後モデルが存在しないのは評価に含めない（）
                        pass
                    else:
                        #前後のモデルのスコア(今回から±１にした)===================================
                        #diff_m2_row=df[df['number_i']==(row['number_i']-2)]
                        diff_m1_row=df[df['number_i']==(row['number_i']-1)]
                        diff_p1_row=df[df['number_i']==(row['number_i']+1)]
                        #diff_p2_row=df[df['number_i']==(row['number_i']+2)]
                        #diff_models=[diff_m2_row,diff_m1_row,diff_p1_row,diff_p2_row]
                        diff_models=[diff_m1_row,diff_p1_row]
                        #===================================================
                        if ((row['利益率']>gain_th) and (row['購買的中率']>buy_accuracy_th) and (row['的中数']>=num_tp_th)):#初めに真ん中のモデルの性能評価
                            #前後モデルの評価
                            flag_arr=[0,0]
                            for index in range(len(flag_arr)):

                                diff_model=diff_models[index].iloc[0,:]
                                if ((diff_model['利益率']>diff_gain_th) and (diff_model['購買的中率']>diff_buy_accuracy_th) and (diff_model['的中数']>=diff_num_tp_th)):#前後モデルの性能評価
                                    flag_arr[index]=1
                                else:
                                    pass
                            #if sum(flag_arr)==2:
                            if sum(flag_arr)>=1:
                                model_para_df=model_para_df.append(row)

                        else:
                            pass
    dir_path = "../bot_database/{place_name}/model_score_{place_name}/good_model/good_model_{place_name}_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_para_df.to_csv(dir_path, encoding='utf_8_sig')

def use_model_para(good_model_df,place_name,version):#実査に使用するのにどのモデルが最適か判別するパラメータを作成する関数(製作中)
    use_model_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数','的中数_std','利益率_std','score'])
    print(place_name)
    for com in sorted(good_model_df['target_com'].value_counts().index):
        target_com_df=good_model_df[good_model_df['target_com']==com]
        #sc = StandardScaler()
        #sc =sc.fit(target_com_df['的中数'].values)
        target_com_df['的中数_std']=scipy.stats.zscore(target_com_df['的中数'].values)
        #target_com_df['的中数_std']=sc.transform(target_com_df['的中数'].values)
        target_com_df['利益率_std']=scipy.stats.zscore(target_com_df['利益率'].values)

        #target_com_df['利益率_std']=sc.fit_transform(target_com_df['利益率'].values)
        target_com_df['score']=target_com_df['的中数_std']+target_com_df['利益率_std']
        if len(target_com_df)>=3:
            target_com_df=target_com_df[target_com_df['利益率_std']>0]#なるべく利益率を重視したモデル、偏差の中央以下の基準の利益率のモデルは候補に入れない。
            use_model_df_row=target_com_df[target_com_df['score']==target_com_df['score'].max()]
        elif len(target_com_df)==2:#モデルが二つしかなかった時、最良のモデルが二つできてまう。
            use_model_df_row=target_com_df[target_com_df['利益率']==target_com_df['利益率'].max()]#そんなときは利益率で選ぶ
        elif len(target_com_df)==1:#モデルが1つしかなかった時、そのまま代入
            use_model_df_row=target_com_df
        use_model_df=pd.concat([use_model_df, use_model_df_row])

    dir_path = "../bot_database/{place_name}/model_score_{place_name}/use_model/use_model_{place_name}_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    use_model_df.to_csv(dir_path, encoding='utf_8_sig')
    return None


def save_model(use_model_df,place_name,version):#実査に使用するのにどのモデルが最適か判別するパラメータを作成する関数(製作中)
    use_model_df=pd.DataFrame(columns=['target_com','depth','target_per','threshold','総収益', '投資金額','出現数','購買予測数','利益率','購買的中率','的中数','的中数_std','利益率_std','score'])
    print(place_name)
    for com in sorted(good_model_df['target_com'].value_counts().index):
        target_com_df=good_model_df[good_model_df['target_com']==com]
        #sc = StandardScaler()
        #sc =sc.fit(target_com_df['的中数'].values)
        target_com_df['的中数_std']=scipy.stats.zscore(target_com_df['的中数'].values)
        #target_com_df['的中数_std']=sc.transform(target_com_df['的中数'].values)
        target_com_df['利益率_std']=scipy.stats.zscore(target_com_df['利益率'].values)

        #target_com_df['利益率_std']=sc.fit_transform(target_com_df['利益率'].values)
        target_com_df['score']=target_com_df['的中数_std']+target_com_df['利益率_std']
        if len(target_com_df)>=3:
            target_com_df=target_com_df[target_com_df['利益率_std']>0]#なるべく利益率を重視したモデル、偏差の中央以下の基準の利益率のモデルは候補に入れない。
            use_model_df_row=target_com_df[target_com_df['score']==target_com_df['score'].max()]
        elif len(target_com_df)==2:#モデルが二つしかなかった時、最良のモデルが二つできてまう。
            use_model_df_row=target_com_df[target_com_df['利益率']==target_com_df['利益率'].max()]#そんなときは利益率で選ぶ
        elif len(target_com_df)==1:#モデルが1つしかなかった時、そのまま代入
            use_model_df_row=target_com_df
        use_model_df=pd.concat([use_model_df, use_model_df_row])

    dir_path = "../bot_database/{place_name}/model_score_{place_name}/use_model/use_model_{place_name}_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    use_model_df.to_csv(dir_path, encoding='utf_8_sig')
    return None


def ym_analysis(result_base_df,use_model_df,place_name,version,year):
    #渡した会場と年ごとに、今あるモデルっを使った時の利益率や的中数のシミュレーションを行う関数
    #探査結果から学習したモデルを保存する関数、
    print(place_name)
    #==============================================================================
    #学習関数で場所ごとにバージョンに対応した学習データを作る
    result_df=making.data_making_clustar(result_base_df)
    #==============================================================================
    #日付データの加工
    date_df=making.get_event_info(result_base_df)

    #学習データの切り分け
    test_df = result_df[result_df['year']==year]#指定の年のデータを検証用データに。
    date_df = date_df[date_df['year']==year]#指定の年のデータを検証用データに。
    #学習データを切り分けたらyearはいらないから削除する
    test_df =test_df.drop('year',axis=1)

    money_col=test_df['money']#配当金情報の削除
    test_df=test_df.drop('money',axis=1)
    result_col=test_df['result_com']#着の組み合わせ
    test_df=test_df.drop('result_com',axis=1)

    #XGboostのデータ型に変換する
    target_x_test=test_df.drop('money',axis=1).copy()
    target_x_test=target_x_test.drop('result_com',axis=1)
    target_y_test=result_col
    test = xgb.DMatrix(target_x_test, label=target_y_test)#実際に使った時の利益率の算出用
    #=====================================================================================
    #=====================================================================================
    models_dict={}
    for index , para_row in use_model_df.iterrows():#use_modelのパラメータシートを参考にモデルをディクショナリに格納
        result_com=int(para_row['target_com'])
        depth=int(para_row['depth'])
        target_per=int(para_row['target_per'])
        th=float(para_row['threshold'])
        pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/com{com}_{depth}_{target_per}_{th}_{place_name}.sav".format(place_name=place_name,com=result_com,depth=depth,target_per=target_per,th=th)#モデルを保存

        models_dict[para_row['target_com']]=pickle.load(open(pickle_path, 'rb'))#組番号に対応したモデルを格納

    pred_df=pd.DataFrame(columns=["pred_{}".format(com) for com in use_model_df['target_com'].values])
    for com, model in models_dict.items():
        pred_df['pred_{}']=model.predict(test)

    #レースの情報を戻す。
    pred_df['result_com']=result_col

    date_cols=['date','year','month','day','num_date','range_date','season']
    for col in date_cols:#分析用に日付のデータも戻す。
        pred_df[col]=date_df[col]
    pred_flags=[]#正答のカラムを作るための結果と予測の比較結果を持ったリスト(正解:1  外れ:0)
    num_preds=[]#予測の個数
    target_coms=[int(com) for com in use_model_df['target_com'].values]#モデルが存在するcom

    for index,row in pred_df.iterrows():
        result=row['result_com']
        flag=0
        pred_count=0#予測した数
        for com_num in target_coms:
            if 1==row['pred_{}'.format(com_num)]:#発生あり予測があるか
                pred_count+=1
                if result==com_num:#予測があっているか
                    flag=1
                else:
                    pass
            else:
                pass
        pred_flags.append(flag)
        num_preds.append(pred_count)#レース内の予測数を持ったカラム
    pred_result_df=pred_df.copy()
    pred_result_df['right_pred']=pred_flags



    pred_result_df['num_pred']=num_preds
    pred_result_df['money']=money_col
    num_pred_df=pd.DataFrame(columns=['num_pred'])
    for month_num in np.arange(1,13,1):#月の番号:
        sp_month_df=pred_result_df[pred_result_df['month']==month_num]#月のデータ
        num_pred_df.loc['{}月'.format(month_num)]=sp_month_df['num_pred'].sum()

    #正しく判断できたレースのみを残す（ほんとは投票を行ったものの数農地の割合のほうがよさそうだけど・・）
    #グラフをP描画する用のdfを作成する。
    right_df=pred_result_df[pred_result_df['right_pred']==1]
    month_graph_df=pd.DataFrame(columns=target_coms)
    for month_num in np.arange(1,13,1):#月の番号
        sp_month_df=right_df[right_df['month']==month_num]#月のデータ
        append_s=sp_month_df['result_com'].value_counts()
        month_graph_df.loc['{}月'.format(month_num)]=append_s

    money_df=pd.DataFrame(columns=['money'])
    for month_num in np.arange(1,13,1):#月の番号:
        sp_month_df=right_df[right_df['month']==month_num]#月のデータ
        money_df.loc['{}月'.format(month_num)]=sp_month_df['money'].sum()


    month_graph_df=month_graph_df.fillna(0)
    #============================================================
    hit_sum_arr=[]
    for index,row in month_graph_df.iterrows():
        hit_sum=0
        for com in target_coms:
            hit_sum+=row[target_coms]
        hit_sum_arr.append(hit_sum)


    month_graph_df['sum']=hit_sum_arr

    month_graph_df['月']=month_graph_df.index
    month_graph_df=month_graph_df.fillna(0)

    month_graph_df=pd.concat([month_graph_df, num_pred_df], axis=1)#月の購買数をその月的中枢に結合
    month_graph_df=pd.concat([month_graph_df, money_df], axis=1)#月の総収益をその月に結合
    month_graph_df['gain']=month_graph_df['money']/(month_graph_df['num_pred']*100)*100#利益率
    month_graph_df['hit_per']=(month_graph_df['sum']/month_graph_df['num_pred'])*100#的中率
    print('利益率平均',(month_graph_df['money'].sum()/(month_graph_df['num_pred'].sum()*100)))#利益率平均(month_graph_df['money'].sum()/(month_graph_df['num_pred'].sum()*100))#利益率平均
    dir_path="../bot_database/{place_name}/model_analysis_{place_name}/{year}_analysis_{version}.csv".format(place_name=place_name,year=year,version=version)#モデルを保存

    month_graph_df.to_csv(dir_path,encoding='utf-8')
    display(month_graph_df)

    #グラフの描画
    X_col='月'
    Y_cols=[str(com) for com in use_model_df['target_com'].values]
    for col in month_graph_df.columns:
        month_graph_df=month_graph_df.rename(columns={col: "{}".format(col)})
    graph.stack_bar(X_col, Y_cols,month_graph_df)
    #print(month_graph_df['sum'].sum())


def save_clustar_model(result_base_df):#クラスタリングあり、モータ番号、艇番号なし


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
        pickle_path="../bot_database/{place_name}/model_pickle_{place_name}/clustering_{place_name}_num_{num_cluster}.sav".format(place_name=place_name,com=result_com,depth=depth,target_per=target_per,th=th,num_cluster=num_cluster)#モデルを保存
        pickle.dump(Km, open(pickle_path, "wb"))#モデルの保存
