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
