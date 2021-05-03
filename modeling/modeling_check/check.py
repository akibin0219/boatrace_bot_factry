#以下pickle版================================================================================================
#以下pickle版================================================================================================
#以下pickle版================================================================================================
#以下pickle版================================================================================================
def pickle_check_V2_1_2(result_base_df,use_model_df,place_name,version):#pickleを使った時の予測内容のチェックをする
def pickle_check_V2_1_2(result_base_df,use_model_df,place_name,version):#pickleを使った時の予測内容のチェックをする
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
    pred_concat_df=pd.DataFrame(columns=use_model_df['target_com'].values,index=test_df.index)#予測データをまとめて持つdf
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

        #======================================================================================
        #===============================================================================
        #学習データのラベル変換==========================================================
        result_train_df=train_df.copy()
        result_train_df=trans_result_com(result_com,result_train_df)

        result_test_df=test_df.copy()
        result_test_df=trans_result_com(result_com,result_test_df)

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
        #test = xgb.DMatrix(target_x_test, label=target_y_test)#実際に使った時の利益率の算出用
        test = xgb.DMatrix(target_x_test)#実際に使った時の利益率の算出用
    #     #xgb.config_context(verbosity=0)
    #     param = {'max_depth': depth, #パラメータの設定
    #                      #'eta': 1.8,
    #                      #'eta': 0.8,
    #                      'eta': 1.3,
    #                      #'eta': 0.2,
    #                      #'objective': 'binary:hinge',
    #                      'objective': 'binary:logistic',#確率で出力
    #                      'eval_metric': 'logloss',
    #                      'verbosity':0,
    #                      'subsample':0.8,
    #                      'nthread':10,
    #                      'gpu_id':0,
    #                      'seed':7,
    #                      'tree_method':'gpu_hist'
    #                     }
    #     evallist = [(valid, 'eval'), (train, 'train')]#学習時にバリデーションを監視するデータの指定。
    #     num_round = 400

        pickle_path="check_pickle/{place_name}/com{com}_{depth}_{target_per}_{th}_{place_name}.sav".format(place_name=place_name,com=result_com,depth=depth,target_per=target_per,th=th)#モデルのdirs
        bst = pickle.load(open(pickle_path, 'rb'))
        # 未知データに対する予測値
        #predict_y_test = RF.predict(target_x_test)
        predict_y_test=bst.predict(test)

        #[1]の正答率を見る
        pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                                  , 'trans_result':target_y_test})
        trans_df=pred_th_trans(pred_test_df,th)
        #num_1=len(trans_df[trans_df['test']==1])
        count=0

        #/////収益計算の項
        trans_df['money']=test_money
        #trans_df['trans_result']=target_y_test
        trans_df['true_result']=test_df['result_com']

        #/////


        #収益計算部分======================================
        #追加　配当金の情報も考慮する。
        result_gain_base_df=calc_gain(trans_df)
        dir_path = "check_csv/{place_name}/pred/check_pred_pickle_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
        result_gain_base_df.to_csv(dir_path, encoding='utf_8_sig')

        pred_concat_df[result_com]=trans_df['trans_result'].values#組の予測を結合

        #scoreのseriesに情報書き込み==================
        model_score_s['threshold']=th
        model_score_s['総収益']=result_gain_base_df["gain"].sum()
        model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
        model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
        model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
        model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
        model_score_s['的中数']=result_gain_base_df['hit'].sum()
        model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
        model_score_df=model_score_df.append(model_score_s,ignore_index=True)

    #モデルの「スコアを保存
    #model_score_df.to_csv('{}_model_score.csv'.format(place), encoding='utf_8_sig')
    dir_path = "check_csv/{place_name}/check_pickle_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')

    dir_path = "check_csv/{place_name}/pred/pred_pickle_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#予測の書き込み
    pred_concat_df.to_csv(dir_path, encoding='utf_8_sig')

    return None


#以下train版================================================================================================
#以下train版================================================================================================
#以下train版================================================================================================
#以下train版================================================================================================
def train_check_V2_1_2(result_base_df,use_model_df,place_name,version):
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
    pred_concat_df=pd.DataFrame(columns=use_model_df['target_com'].values,index=test_df.index)#予測データをまとめて持つdf
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
        #[1]の正答率を見る
        pred_test_df=pd.DataFrame({'pred_proba':predict_y_test#確率分布での出力
                                  , 'trans_result':target_y_test})
        trans_df=pred_th_trans(pred_test_df,th)
        #num_1=len(trans_df[trans_df['test']==1])
        count=0

        #/////収益計算の項
        trans_df['money']=test_money
        #trans_df['trans_result']=target_y_test
        trans_df['true_result']=test_df['result_com']

        #/////


        #収益計算部分======================================
        #追加　配当金の情報も考慮する。
        result_gain_base_df=calc_gain(trans_df)

        dir_path = "check_csv/{place_name}/pred/check_pred_train_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
        result_gain_base_df.to_csv(dir_path, encoding='utf_8_sig')

        pred_concat_df[result_com]=trans_df['trans_result'].values#組の予測を結合

        #scoreのseriesに情報書き込み==================
        model_score_s['threshold']=th
        model_score_s['総収益']=result_gain_base_df["gain"].sum()
        model_score_s['投資金額']=100*result_gain_base_df["pred"].sum()
        model_score_s['出現数']=result_gain_base_df['trans_result'].sum()
        model_score_s['購買予測数']=result_gain_base_df['pred'].sum()
        model_score_s['利益率']=(model_score_s['総収益']/model_score_s['投資金額'])*100
        model_score_s['的中数']=result_gain_base_df['hit'].sum()
        model_score_s['購買的中率']=(model_score_s['的中数']/ model_score_s['購買予測数'])*100
        model_score_df=model_score_df.append(model_score_s,ignore_index=True)
    #モデルの「スコアを保存


    dir_path = "check_csv/{place_name}/check_train_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#作成したデータの書き込み先#使用するデータの読み込み
    model_score_df.to_csv(dir_path, encoding='utf_8_sig')

    dir_path = "check_csv/{place_name}/pred/pred_train_{place_name}_model_score_{V}.csv".format(place_name=place_name,V=version)#予測の書き込み
    pred_concat_df.to_csv(dir_path, encoding='utf_8_sig')
    return None
