
#初めの製作時================================================================================================
#初めの製作時================================================================================================
#初めの製作時================================================================================================
#初めの製作時================================================================================================



def regulation_pred_proba_scale(pred_df,target_date):
    #購買対象のレースの予測値を持ったdfと，その日の日付を渡す
    #渡したdfを昨年の予測値でーたにまぜて，予測値の差が見えやすいようにスケーリングする．関数

    #year:予測対象のレースが存在する年
    pred_df_len=len(pred_df)
    year=target_date.year#予測対象のレースの年情報だけ切り抜く
    sample_year=year-1#一年前のprobaのデータに混ぜてスケーリングを行う
    sample_proba_df=pd.read_csv('test_csv/proba_get_use_{}.csv'.format(sample_year))
    #sample_proba_df=pd.read_csv('test_csv/proba_get_use_{}.csv'.format(sample_year))
    num_race=np.arange(1,12)

    sample_proba_df=sample_proba_df.loc[:, sample_proba_df.columns.str.contains('pred')]#予測に関する列のみを抽出
    concat_target_proba_df=pd.concat([sample_proba_df, pred_df], axis=0)
    #probaをスケーリングして最大値1,最小値0にスケーリングする（pickleのモデルを使ってスケール変換機の保存を行う）
    for col in sample_proba_df.columns:
        concat_target_proba_df[col]=preprocessing.minmax_scale(concat_target_proba_df[col].values)
    #trans_target_proba_df=concat_target_proba_df[concat_target_proba_df['date']==target_date].copy()
    trans_target_proba_df=concat_target_proba_df[len(concat_target_proba_df)-pred_df_len:].copy()
    #trans_proba_df=trans_proba_df.mask(trans_proba_df>=0.5,1)
    th=0.8#中心点が0.5とは限らないっぽい，なんとなく見ながら変更をしていく
    trans_target_proba_df=trans_target_proba_df.mask(trans_target_proba_df<th,0)#データの中心は変わらないので，0.5未満は購買を行わない
    trans_target_proba_df=((trans_target_proba_df-th)*2)
    #trans_target_proba_df=(((trans_target_proba_df-th)*2)*10000)昨|日の製作時はここで係数をかけたが，実装時にはやらない.
    trans_target_proba_df=trans_target_proba_df.mask(trans_target_proba_df<=0,0)#上記の計算式だと購買を行わないものはみんな-1000となるので０に置換する
    return trans_target_proba_df

def regulation_test(regulation_df,bet_coefficient):
    bet_regulation_df=regulation_df.copy()
    bet_regulation_df=regulation_df.set_axis(regulation_df.columns+"_bet",axis=1).copy()#購買金額に関連する列とわかるように名前を振りなおす
    #bet_regulation_df=((bet_regulation_df)*10000)
    bet_regulation_df=((bet_regulation_df)*bet_coefficient)
    #bet_proba_df=bet_proba_df.mask(bet_proba_df<=0,0)#上記の計算式だと購買を行わないものはみんな-1000となるので０に置換する
    bet_flag_df=regulation_df.copy()
    #bet_flag_df=bet_flag_df.mask(bet_flag_df>=th,1).copy()#データの中心は変わらない,かつ中心以上により購買を行ったものにはフラグ付けを行う
    bet_flag_df=bet_flag_df.set_axis(bet_flag_df.columns+"_buy_flag",axis=1)#購買フラグに関連する列とわかるように名前を振りなおす
    bet_flag_df=bet_flag_df.mask(bet_flag_df>0,1).copy()#データの中心は変わらない,かつ中心以上により購買を行ったものにはフラグ付けを行う
    proba_bet_flag_df=pd.concat([regulation_df,bet_regulation_df],axis=1)
    proba_bet_flag_df=pd.concat([proba_bet_flag_df,bet_flag_df],axis=1)
    #あたったレースにフラグを付ける＆獲得できた配当金の計算（レース単位でユニーク．前に出てきたものとしょりは　似ているが同じではない．）
    return proba_bet_flag_df



def pred_race_former_asiya_proba(date):
    race_df=pd.DataFrame(index=[], columns=[])
    for i in range(12):
        rno=i+1
        #まず初めに１ページの情報を抜き出す機能
        url='http://www.boatrace.jp/owpc/pc/race/racelist?rno={}&jcd=21&hd={}'.format(rno,date)
        response=requests.get(url)#対象のURLをget
        response.encoding = response.apparent_encoding
        start_page=bs4(response.text, 'html.parser')
        racers_div=start_page.select_one(".is-tableFixed__3rdadd")
        racers_sep_row=racers_div.find_all('tbody')
        index=0
        race_racers_data=[0]*6
        if len(racers_sep_row)==6:
            for racer_html in racers_sep_row:
                racer_row_ex_td=racer_html.find_all('td')
                #選手の登録ID
                racer_ID_div=racer_row_ex_td[2].find_all('div')
                racer_ID_div=racer_ID_div[0]
                racer_ID_div_txt=racer_ID_div.text
                start=(racer_ID_div_txt.find('/'))-35
                end=(racer_ID_div_txt.find('/')-31)
                #racer_ID=racer_ID_div_txt[start:end]#選手の登録ID
                racer_ID=racer_ID_div_txt.split('\n')[0]

                #選手のモータ番号
                racer_moter_td=racer_row_ex_td[6]
                racer_moter_td_text=racer_moter_td.text
                #racer_moter_split=racer_moter_td_text.split('\r')
                racer_moter_split=racer_moter_td_text.split('\n')
                #racer_moter_id=racer_moter_split[1][-2:]
                racer_moter_id=racer_moter_split[0]
                #選手のボート番号
                racer_boat_td=racer_row_ex_td[7]
                racer_boat_td_text=racer_boat_td.text
    #             racer_boat_split=racer_boat_td_text.split('\r')
    #             racer_boat_id=racer_boat_split[1][-2:]
                racer_boat_split=racer_boat_td_text.split('\n')
                racer_boat_id=racer_boat_split[0]

                racer_data=[racer_ID,racer_moter_id,racer_boat_id]#選手固有のデータを持ったリスト

                race_racers_data[index]=racer_data
                index+=1
        else:
            print('欠場選手等の例外が発生していないか確認してください')
        #race_racers_dataを一行に変換する
        race_row=pd.Series(index=[])#一レースの情報を一行に変換したもの
        race_num=rno#ここにはレース番号を入れる変数の値を代入
        race_row['number_race']=race_num
        for i in range(len(race_racers_data)):
            race_row['racer_{}_ID'.format(i+1)]=int(race_racers_data[i][0])
            race_row['racer_{}_mo'.format(i+1)]=int(race_racers_data[i][1])
            race_row['racer_{}_bo'.format(i+1)]=int(race_racers_data[i][2])
        race_df=race_df.append(race_row,ignore_index=True)


    ### 学習データと保存してある選手パラメータを結合する
    ### 学習データと保存してある選手パラメータを結合する
    ### 学習データと保存してある選手パラメータを結合する

    #使用するファイルの定義
    para_file='21'
    para_file_path="../../bot_database/racer_para/{}/{}.csv".format(para_file,para_file)
    #/content/drive/My Drive/boatrace_BOT_making/pred_tool/racer_pala_box/20.csv
    #result_file_path="asiya_result_csv/asiya_result_20{0}.csv".format(year)
    #write_path="/content/drive/My Drive/pred_tool/asiya/pred_data/{0}_asiya.csv".format(date)
    #/////////////////////////////////////////////以下データフレームの作成
    para_df=pd.read_csv(para_file_path)
    para_df=para_df.drop(["Unnamed: 0"],axis=1)#csvファイルについている名無しの列を削除
    #出力用データフレーム
    #pred_race_df=pd.DataFrame(columns=['result_com','number_race','racer_1_ID','racer_2_ID','racer_3_ID','racer_4_ID','racer_5_ID','racer_6_ID','racer_1_rank','racer_1_male','racer_1_age','racer_1_doub','racer_1_ave_st','racer_2_rank','racer_2_male','racer_2_age','racer_2_doub','racer_2_ave_st','racer_3_rank','racer_3_male','racer_3_age','racer_3_doub','racer_3_ave_st','racer_4_rank','racer_4_male','racer_4_age','racer_4_doub','racer_4_ave_st','racer_5_rank','racer_5_male','racer_5_age','racer_5_doub','racer_5_ave_st','racer_6_rank','racer_6_male','racer_6_age','racer_6_doub','racer_6_ave_st'])
    pred_race_df=pd.DataFrame(columns=[],index=[])
    for index,series in race_df.iterrows():
        add_df=pd.DataFrame(columns=['number_race','racer_1_ID','racer_2_ID','racer_3_ID','racer_4_ID','racer_5_ID','racer_6_ID','racer_1_rank','racer_1_male','racer_1_age','racer_1_doub','racer_1_ave_st','racer_2_rank','racer_2_male','racer_2_age','racer_2_doub','racer_2_ave_st','racer_3_rank','racer_3_male','racer_3_age','racer_3_doub','racer_3_ave_st','racer_4_rank','racer_4_male','racer_4_age','racer_4_doub','racer_4_ave_st','racer_5_rank','racer_5_male','racer_5_age','racer_5_doub','racer_5_ave_st','racer_6_rank','racer_6_male','racer_6_age','racer_6_doub','racer_6_ave_st'])
        #///////////////////////////////////////レースに出ているレーサーの成績を検索＆取得
        ID_1=series['racer_1_ID']
        ID_2=series['racer_2_ID']
        ID_3=series['racer_3_ID']
        ID_4=series['racer_4_ID']
        ID_5=series['racer_5_ID']
        ID_6=series['racer_6_ID']
        racer_1_df=para_df[para_df['racer_ID']==ID_1]
        racer_2_df=para_df[para_df['racer_ID']==ID_2]
        racer_3_df=para_df[para_df['racer_ID']==ID_3]
        racer_4_df=para_df[para_df['racer_ID']==ID_4]
        racer_5_df=para_df[para_df['racer_ID']==ID_5]
        racer_6_df=para_df[para_df['racer_ID']==ID_6]
        if len(racer_1_df)==1:
            pass
        else:
            racer_1_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_2_df)==1:
            pass
        else:
            racer_2_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_3_df)==1:
            pass
        else:
            racer_3_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_4_df)==1:
            pass
        else:
            racer_4_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        if len(racer_5_df)==1:
            pass
        else:
            racer_5_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')


        if len(racer_6_df)==1:
            pass
        else:
            racer_6_df=para_df[(para_df['racer_doub_win']==0.00) & (para_df['racer_ave_st_time']==0.00)]
            print('CAREFULL!!!!    NOT FOUND RACER ')

        #追加していくデータフレームを作成
        add_df= pd.DataFrame({'number_race':series['number_race'],
                              'racer_1_ID':series['racer_1_ID'],
                                'racer_2_ID':series['racer_2_ID'],
                                'racer_3_ID':series['racer_3_ID'],
                                'racer_4_ID':series['racer_4_ID'],
                                'racer_5_ID':series['racer_5_ID'],
                                'racer_6_ID':series['racer_6_ID'],
                                'racer_1_bo':series['racer_1_bo'],
                                'racer_1_mo':series['racer_1_mo'],
                                'racer_2_bo':series['racer_2_bo'],
                                'racer_2_mo':series['racer_2_mo'],
                                'racer_3_bo':series['racer_3_bo'],
                                'racer_3_mo':series['racer_3_mo'],
                                'racer_4_bo':series['racer_4_bo'],
                                'racer_4_mo':series['racer_4_mo'],
                                'racer_5_bo':series['racer_5_bo'],
                                'racer_5_mo':series['racer_5_mo'],
                                'racer_6_bo':series['racer_6_bo'],
                                'racer_6_mo':series['racer_6_mo'],
                                'racer_1_age':racer_1_df.iat[0,3],
                                'racer_1_ave_st':racer_1_df.iat[0,5],
                                'racer_1_doub':racer_1_df.iat[0,4],
                                'racer_1_rank':racer_1_df.iat[0,1],
                                'racer_1_male':racer_1_df.iat[0,2],

                                'racer_2_age':racer_2_df.iat[0,3],
                                'racer_2_ave_st':racer_2_df.iat[0,5],
                                'racer_2_doub':racer_2_df.iat[0,4],
                                'racer_2_rank':racer_2_df.iat[0,1],
                                'racer_2_male':racer_2_df.iat[0,2],

                                'racer_3_age':racer_3_df.iat[0,3],
                                'racer_3_ave_st':racer_3_df.iat[0,5],
                                'racer_3_doub':racer_3_df.iat[0,4],
                                'racer_3_rank':racer_3_df.iat[0,1],
                                'racer_3_male':racer_3_df.iat[0,2],

                                'racer_4_age':racer_4_df.iat[0,3],
                                'racer_4_ave_st':racer_4_df.iat[0,5],
                                'racer_4_doub':racer_4_df.iat[0,4],
                                'racer_4_rank':racer_4_df.iat[0,1],
                                'racer_4_male':racer_4_df.iat[0,2],


                                'racer_5_age':racer_5_df.iat[0,3],
                                'racer_5_ave_st':racer_5_df.iat[0,5],
                                'racer_5_doub':racer_5_df.iat[0,4],
                                'racer_5_rank':racer_5_df.iat[0,1],
                                'racer_5_male':racer_5_df.iat[0,2],

                                'racer_6_age':racer_6_df.iat[0,3],
                                'racer_6_ave_st':racer_6_df.iat[0,5],
                                'racer_6_doub':racer_6_df.iat[0,4],
                                'racer_6_rank':racer_6_df.iat[0,1],
                                'racer_6_male':racer_6_df.iat[0,2] }, index=[''])
        #//////////////////////////////データフレームにadd_dfを追加していく。
        pred_race_df=pred_race_df.append(add_df)
    #pred_race_df.to_csv('/content/drive/My Drive/pred_tool/asiya/start_list/{}_starts_asiya.csv'.format(date))
    model_df=making_pred_df(pred_race_df)

    #保存してあるモデルを読み込む(ついでに予測も行う)
    #保存してあるモデルを読み込む(ついでに予測も行う)
    #保存してあるモデルを読み込む(ついでに予測も行う)
    #保存してあるモデルを読み込む(ついでに予測も行う)
    num_race=np.arange(1,13,1)
    model3 = pickle.load(open('former_pickle_data/model_com3_dep6_per121.sav', 'rb'))
    model4 = pickle.load(open('former_pickle_data/model_com4_dep7_per131.sav', 'rb'))
    model5 = pickle.load(open('former_pickle_data/model_com5_dep8_per122.sav', 'rb'))
    model7 = pickle.load(open('former_pickle_data/model_com7_dep7_per146.sav', 'rb'))
    model13 = pickle.load(open('former_pickle_data/model_com13_dep6_per115.sav', 'rb'))
    model14= pickle.load(open('former_pickle_data/model_com14_dep4_per123.sav', 'rb'))
    model3_pred=model3.predict_proba(model_df)
    model4_pred=model4.predict_proba(model_df)
    model5_pred=model5.predict_proba(model_df)
    model7_pred=model7.predict_proba(model_df)
    model13_pred=model13.predict_proba(model_df)
    model14_pred=model14.predict_proba(model_df)
    pred_3_df=pd.DataFrame({'num_race':num_race,
                            'pred_3':[pred[1] for pred in model3_pred],
                            'pred_4':[pred[1] for pred in model4_pred],
                            'pred_5':[pred[1] for pred in model5_pred],
                            'pred_7':[pred[1] for pred in model7_pred],
                            'pred_13':[pred[1] for pred in model13_pred],
                            'pred_14':[pred[1] for pred in model14_pred]
                            })
    #print("pred_day_url",url)
    return pred_3_df


def making_pred_df(df):
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
    result_df=result_df_dummie



  #クラスタリング
  #分けてみるクラスタの数は[8,10]の2個
  #cluster_target_df　　trainのデータからリザルトと配当金を取り除いたもの
    target_num_cluster=[8,10]
  #test_clustaring_df=train_has_PCA_df
    clustar_target_df=result_df
    clustaring_df=clustar_target_df
    for num_cluster in target_num_cluster:
        pred = KMeans(n_clusters=num_cluster).fit_predict(clustar_target_df)
        clustaring_df['num={}'.format(num_cluster)]=pred

    model_df=clustaring_df

    return model_df


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
