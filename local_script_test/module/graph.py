import datetime as dt
import pandas.io.sql as sqlio
import sys, os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import codecs
from matplotlib.ticker import FuncFormatter
# 出力画像設定
FIGSIZE = (20, 8)
DPI = 200
FONTSIZE = 16
mpl.rcParams['font.family'] = 'VL Gothic'

#分布に変換する関数=====================================
def output_df(X_name,Y_name,target_df, distribusion_list_y, distribusion_list_x):
    """
    対象のDFのリピータ率やオッズ比の分布について購買回数ごとに可視化する
    """
    """
    [EX]
    Y_name='リピータ率'
    X_name='リピートした顧客数'
    distribusion_list_y = [0.5, 1.0, 1.5, 2, 2.5, 3, 3.5]　　リピータ率閾値リスト
    distribusion_list_x = [10,50, 100, 500, 1000, 3000,5000,10000,50000] 顧客数閾値リスト
    """
    columns = list()
    start = 0
    for c in distribusion_list_x:
        columns.append('{} ~ {}'.format(start, c))
        start = c

    df_tmp = pd.DataFrame(columns=columns)

    # 初期条件
    left_r = 0 #行

    for r in distribusion_list_y:
        # Y範囲ごとにループ
        s = list()
        #print('行 : {} ~ {}'.format(left_r, r))
        target_df_r = target_df[(left_r<target_df[Y_name])&(target_df[Y_name]<=r)]

        # 初期条件
        left_c = 0 #列
        for c in distribusion_list_x:
            # 購買回数ごとにループ
            #print('列 : {} ~ {}'.format(left_c, c))
            s.append(len(target_df_r[(left_c<target_df_r[X_name])&(target_df_r[X_name]<=c)]))
            left_c = c

        s = pd.Series(s, index=columns, name='{} ~ {}'.format(left_r, r))
        df_tmp = df_tmp.append(s, ignore_index=False)

        left_r = r
    # オッズ比の計算ができない得意顧客でない顧客の購買数が0
    s = list()
    target_df_r = target_df[target_df[Y_name]==float('inf')]
    for c in distribusion_list_x:
        # 購買回数ごとにループ
        #print('列 : {} ~ {}'.format(left_c, c))
        s.append(len(target_df_r[(left_c<target_df_r[X_name])&(target_df_r[X_name]<=c)]))
        left_c = c

    s = pd.Series(s, index=columns, name='のみ')
    df_tmp = df_tmp.append(s, ignore_index=False)

    return df_tmp
#=================================================================
def value_servay(X_name,Y_name,Y_range,X_range,df):#指定した範囲のオッズ比やリピータ率、購買数を持つvalueを抜き出す関数
    """
    Y_name='リピータ率'
    X_name='リピートした顧客数'
    Y_range=[0.1,0.3]
    X_range=[1000,3000]
    """
    min_Y=Y_range[0]
    max_Y=Y_range[1]
    min_X=X_range[0]
    max_X=X_range[1]

    target_df=df[(min_X<df[X_name])&(df[X_name]<=max_X)]#X
    target_df=target_df[(min_Y<target_df[Y_name])&(target_df[Y_name]<=max_Y)]#Y

    return target_df
#=================================================================
#元のdf1に指定したカラムの値の全体に対する割合と、堆積比率を持たせる
def ratio(target_col,df):
    #累積比率を算出
    """
    [EX]
    target_col='購買回数'
    """


    rat = 0
    cumulative_ratio = list()
    ratio = list()
    Y=df[target_col]
    for i in Y:
        rat += (i/sum(Y))*100
        ratio.append((i/sum(Y))*100)
        cumulative_ratio.append(rat)
    ret_df=df.copy()
    ret_df['比率']=ratio
    ret_df['累積比率']=cumulative_ratio
    return ret_df
#=================================================================
#=================================================================
#累積比率を求めたdfから、パレート図を出力
def parat(X_col,Y_col,df,out_pic=False,X_rotation=90):
    ret_df=df
    X=ret_df[X_col]
    Y=ret_df[Y_col]
    cumulative_ratio=ret_df['累積比率']

    fig, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax1.bar(X, Y, label=Y_col)
    plt.xticks(rotation=X_rotation)
    #軸を複製
    ax2 = ax1.twinx()
    #折れ線グラフでプロット
    ax2.plot(np.arange(len(X)), cumulative_ratio, color='red', label='累積比率')
    ax2.set_ylabel('Y2', rotation=0)

    #軸ラベルの設定
    ax1.set_ylabel(Y_col,rotation=90)
    ax1.set_xlabel(X_col)
    ax2.set_ylabel('比率の累計[%]', rotation=90)
    # 凡例情報を取得する
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='center', bbox_to_anchor=(0.5, 1.12))
    if out_pic==False:
    	pass
    else:
        plt.savefig('parat.png', bbox_inches='tight', transparent=True)

    return None
#====================================================================
#SQLの実行（結果をpd.dataframeにしたいとき）
def execute_sql_todf(sql):
    with psycopg2.connect("host=10.0.1.12 dbname=kaunet port=5432 user=staff password=staff") as conn:
        return pd.read_sql(sql, conn)
#===================================================================
#混合行列を作成する関数
def make_conf(target_df,x_name,y_name):#target_df:予測と正解を1,0で持ったdfを渡す   x_name:混合行列の列名(columns)  y_name(混同行列の行(index))
    pred_cross_df=pd.DataFrame(index=['{}=1'.format(y_name),'{}=0'.format(y_name)], columns=['{}=1'.format(x_name),'{}=0'.format(x_name)])
    pred_cross_df.iat[0,0]=len(target_df[(target_df[x_name]==1)&(target_df[y_name]==1)])
    pred_cross_df.iat[0,1]=len(target_df[(target_df[x_name]==0)&(target_df[y_name]==1)])
    pred_cross_df.iat[1,0]=len(target_df[(target_df[x_name]==1)&(target_df[y_name]==0)])
    pred_cross_df.iat[1,1]=len(target_df[(target_df[x_name]==0)&(target_df[y_name]==0)])
    pred_cross_df['合計']=[pred_cross_df.loc['{}=1'.format(y_name)].sum(),pred_cross_df.loc['{}=0'.format(y_name)].sum()]
    pred_cross_df.loc['合計']=[pred_cross_df['{}=1'.format(x_name)].sum(),pred_cross_df['{}=0'.format(x_name)].sum(),pred_cross_df['合計'].sum()]
    return pred_cross_df
#=======================================================================
#混合行列の各種評価値を算出する関数
def conf_score(pred_cross_df):
    #再現率
    recall=pred_cross_df.iat[0,0]/(pred_cross_df.iat[0,0]+pred_cross_df.iat[0,1])

    #適合率
    precision=pred_cross_df.iat[0,0]/(pred_cross_df.iat[0,0]+pred_cross_df.iat[1,0])

    #的中率
    accuracy=((pred_cross_df.iat[0,0]+pred_cross_df.iat[1,1])/pred_cross_df.iat[2,2])*100
    #F値
    F=2*((recall*precision)/(recall+precision))
    print('再現率:',recall,',   ','適合率:',precision,', ','的中率:',accuracy,'% ,  ','F値:',F)
    return [recall,precision,accuracy,F]
#=====================================================================================
#シンプルな棒グラフの出力
def out_bar(x_col,y_col,df,x_rotation=90,out_pic=False):
    """
    ・x_col  件数を求めるための集約名の入ったカラムの名前(ex): x_col=’月’
    ・y_col  集約名の件数が入ったカラムの名前(ex): y_col=’顧客数
    オプション設定=======================================================
    ・out_pic 作成したグラフを画像として出力するか(ディフォルトはFalse)
　　　　 Trueで[bar.png]が同じ階層に出力される
    ・X_rotarion  X軸の値を何度回転させるか(ディフォルトは90)
    =================================================================
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    width = 0.8

    plt.bar(df[x_col].values,df[y_col].values,tick_label=df[x_col].values, width=width)
    plt.xticks(df[x_col].values,rotation=90)
    #plt.legend(bbox_to_anchor=(0.5, 1.12), ncol=1, frameon=False, loc='center', fontsize=FONTSIZE)
    plt.ylabel(y_col, rotation=x_rotation)
    plt.xlabel(x_col)
    plt.grid(axis='y')
    if out_pic==False:
        pass
    else:
        plt.savefig('bar.png', bbox_inches='tight', transparent=True)
    plt.show()
    return None
#積み上げ棒グラフの作成=====================================================================================
def stack_bar(X_col, Y_cols,df,Y_name=None,rotation=90,colors=None,out_pic=False):
    # 混同行列各数値時系列可視化
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x_label_list=df[X_col].values#X軸のラベル
    bottom = [0]*len(df)
    if colors==None:
        for i in range(len(Y_cols)):
            target_list=df['{}'.format(Y_cols[i])].values
            ax.bar(np.arange(len(target_list)), target_list, bottom=bottom, label=Y_cols[i], alpha=1.0)
            bottom = [i+j for i, j in zip(bottom, target_list)]
    else:
        for i in range(len(Y_cols)):
            target_list=df['{}'.format(Y_cols[i])].values
            ax.bar(np.arange(len(target_list)), target_list, bottom=bottom, color=colors[i],label=Y_cols[i], alpha=1.0)
            bottom = [i+j for i, j in zip(bottom, target_list)]


    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(x_label_list, rotation=rotation, fontsize=FONTSIZE)
    #ax.set_ylim([0,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #ax.yaxis.set_major_formatter(yen_formatter)

    ax.tick_params(axis='y', labelsize=FONTSIZE)
    ax.legend(bbox_to_anchor=(0.5, 1.12), ncol=1, frameon=False, loc='center', fontsize=FONTSIZE)
    #軸ラベルの設定
    if Y_name!=None:
    	ax.set_ylabel(Y_name,rotation=90)
    ax.set_xlabel(X_col)
    ax.grid(axis='y')
    if out_pic==False:
        pass
    else:
        plt.savefig('stack_bar.png', bbox_inches='tight', transparent=True)
    return None
