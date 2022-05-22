create schema proba_test;

/*テーブル作成用のコード（レース単位のものは基本的に「日付，会場名，レース番号でユニーク化」）*/

/*bet時の金額，フラグ関連のテーブル，レース単位でユニーク*/
CREATE TABLE proba_test.bet_log_former_asiya_proba_test_2022(
  place_name VARCHAR ,
  date text,
  num_race INTEGER ,
  racer_1_id INTEGER,
  racer_2_id INTEGER,
  racer_3_id INTEGER,
  racer_4_id INTEGER,
  racer_5_id INTEGER,
  racer_6_id INTEGER,
  proba_3 DOUBLE PRECISION	,
  proba_4 DOUBLE PRECISION	,
  proba_5 DOUBLE PRECISION	,
  proba_7 DOUBLE PRECISION	,
  proba_13 DOUBLE PRECISION	,
  proba_14 DOUBLE PRECISION	,
  pred_3 DOUBLE PRECISION	,
  pred_4 DOUBLE PRECISION	,
  pred_5 DOUBLE PRECISION	,
  pred_7 DOUBLE PRECISION	,
  pred_13 DOUBLE PRECISION	,
  pred_14 DOUBLE PRECISION	,
  bet_3 DOUBLE PRECISION	,
  bet_4 DOUBLE PRECISION	,
  bet_5 DOUBLE PRECISION	,
  bet_7 DOUBLE PRECISION	,
  bet_13 DOUBLE PRECISION	,
  bet_14 DOUBLE PRECISION	,
  buy_flag_3 DOUBLE PRECISION	,
  buy_flag_4 DOUBLE PRECISION	,
  buy_flag_5 DOUBLE PRECISION	,
  buy_flag_7 DOUBLE PRECISION	,
  buy_flag_13 DOUBLE PRECISION	,
  buy_flag_14 DOUBLE PRECISION	,
  total_use INTEGER
);

/*bet,getそれぞれの合計のテーブル，集計用（日付単位でユニーク）*/
CREATE TABLE proba_test.bet_get_log_former_asiya_proba_test_2022(
    place_name VARCHAR ,
    date text ,
    money INTEGER,
    money_type VARCHAR

);

/*get時の結果をまとめておく（自分の要素入っておらず，結果だけが格納されている），bet時と同様にレース単位でユニーク*/
CREATE TABLE proba_test.result_log_former_asiya_proba_test_2022(
    place_name VARCHAR ,
    date text,
    num_race INTEGER ,
    result_coms INTEGER ,
    return_money INTEGER
);

/*ミスった時用のテーブル削除用のクエリ*/
DROP TABLE proba_test.bet_log_former_asiya_proba_test_2022;
DROP TABLE proba_test.bet_get_log_former_asiya_proba_test_2022;
DROP TABLE proba_test.result_log_former_asiya_proba_test_2022;

--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--加工無し用probaを出力したやつの書き込み先ver

CREATE TABLE proba_test.bet_log_former_asiya_proba_check_test_2022(
    place_name VARCHAR ,
    date text,
    num_race INTEGER ,
    racer_1_id INTEGER,
    racer_2_id INTEGER,
    racer_3_id INTEGER,
    racer_4_id INTEGER,
    racer_5_id INTEGER,
    racer_6_id INTEGER,
    proba_3 DOUBLE PRECISION	,
    proba_4 DOUBLE PRECISION	,
    proba_5 DOUBLE PRECISION	,
    proba_7 DOUBLE PRECISION	,
    proba_13 DOUBLE PRECISION	,
    proba_14 DOUBLE PRECISION	,
    pred_3 DOUBLE PRECISION	,
    pred_4 DOUBLE PRECISION	,
    pred_5 DOUBLE PRECISION	,
    pred_7 DOUBLE PRECISION	,
    pred_13 DOUBLE PRECISION	,
    pred_14 DOUBLE PRECISION	,
    bet_3 DOUBLE PRECISION	,
    bet_4 DOUBLE PRECISION	,
    bet_5 DOUBLE PRECISION	,
    bet_7 DOUBLE PRECISION	,
    bet_13 DOUBLE PRECISION	,
    bet_14 DOUBLE PRECISION	,
    buy_flag_3 DOUBLE PRECISION	,
    buy_flag_4 DOUBLE PRECISION	,
    buy_flag_5 DOUBLE PRECISION	,
    buy_flag_7 DOUBLE PRECISION	,
    buy_flag_13 DOUBLE PRECISION	,
    buy_flag_14 DOUBLE PRECISION	,
    total_use INTEGER
);

/*bet,getそれぞれの合計のテーブル，集計用（日付単位でユニーク）*/
CREATE TABLE proba_test.bet_get_log_former_asiya_proba_check_test_2022(
    place_name VARCHAR ,
    date text ,
    money INTEGER,
    money_type VARCHAR

);

/*get時の結果をまとめておく（自分の要素入っておらず，結果だけが格納されている），bet時と同様にレース単位でユニーク*/
CREATE TABLE proba_test.result_log_former_asiya_proba_check_test_2022(
    place_name VARCHAR ,
    date text,
    num_race INTEGER ,
    result_coms INTEGER ,
    return_money INTEGER
);

/*ミスった時用のテーブル削除用のクエリ*/
DROP TABLE proba_test.bet_log_former_asiya_proba_check_test_2022;
DROP TABLE proba_test.bet_get_log_former_asiya_proba_check_test_2022;
DROP TABLE proba_test.result_log_former_asiya_proba_check_test_2022;
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================






--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--th07
CREATE TABLE proba_test.bet_log_former_asiya_proba_test_2022_th07(
  place_name VARCHAR ,
  date text,
  num_race INTEGER ,
  racer_1_id INTEGER,
  racer_2_id INTEGER,
  racer_3_id INTEGER,
  racer_4_id INTEGER,
  racer_5_id INTEGER,
  racer_6_id INTEGER,
  proba_3 DOUBLE PRECISION	,
  proba_4 DOUBLE PRECISION	,
  proba_5 DOUBLE PRECISION	,
  proba_7 DOUBLE PRECISION	,
  proba_13 DOUBLE PRECISION	,
  proba_14 DOUBLE PRECISION	,
  pred_3 DOUBLE PRECISION	,
  pred_4 DOUBLE PRECISION	,
  pred_5 DOUBLE PRECISION	,
  pred_7 DOUBLE PRECISION	,
  pred_13 DOUBLE PRECISION	,
  pred_14 DOUBLE PRECISION	,
  bet_3 DOUBLE PRECISION	,
  bet_4 DOUBLE PRECISION	,
  bet_5 DOUBLE PRECISION	,
  bet_7 DOUBLE PRECISION	,
  bet_13 DOUBLE PRECISION	,
  bet_14 DOUBLE PRECISION	,
  buy_flag_3 DOUBLE PRECISION	,
  buy_flag_4 DOUBLE PRECISION	,
  buy_flag_5 DOUBLE PRECISION	,
  buy_flag_7 DOUBLE PRECISION	,
  buy_flag_13 DOUBLE PRECISION	,
  buy_flag_14 DOUBLE PRECISION	,
  total_use INTEGER
);

/*bet,getそれぞれの合計のテーブル，集計用（日付単位でユニーク）*/
CREATE TABLE proba_test.bet_get_log_former_asiya_proba_test_2022_th07(
    place_name VARCHAR ,
    date text ,
    money INTEGER,
    money_type VARCHAR

);

/*get時の結果をまとめておく（自分の要素入っておらず，結果だけが格納されている），bet時と同様にレース単位でユニーク*/
CREATE TABLE proba_test.result_log_former_asiya_proba_test_2022_th07(
    place_name VARCHAR ,
    date text,
    num_race INTEGER ,
    result_coms INTEGER ,
    return_money INTEGER
);

/*ミスった時用のテーブル削除用のクエリ*/
DROP TABLE proba_test.bet_log_former_asiya_proba_test_2022_th07;
DROP TABLE proba_test.bet_get_log_former_asiya_proba_test_2022_th07;
DROP TABLE proba_test.result_log_former_asiya_proba_test_2022_th07;




--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--===================================================================================
--th09
CREATE TABLE proba_test.bet_log_former_asiya_proba_test_2022_th09(
  place_name VARCHAR ,
  date text,
  num_race INTEGER ,
  racer_1_id INTEGER,
  racer_2_id INTEGER,
  racer_3_id INTEGER,
  racer_4_id INTEGER,
  racer_5_id INTEGER,
  racer_6_id INTEGER,
  proba_3 DOUBLE PRECISION	,
  proba_4 DOUBLE PRECISION	,
  proba_5 DOUBLE PRECISION	,
  proba_7 DOUBLE PRECISION	,
  proba_13 DOUBLE PRECISION	,
  proba_14 DOUBLE PRECISION	,
  pred_3 DOUBLE PRECISION	,
  pred_4 DOUBLE PRECISION	,
  pred_5 DOUBLE PRECISION	,
  pred_7 DOUBLE PRECISION	,
  pred_13 DOUBLE PRECISION	,
  pred_14 DOUBLE PRECISION	,
  bet_3 DOUBLE PRECISION	,
  bet_4 DOUBLE PRECISION	,
  bet_5 DOUBLE PRECISION	,
  bet_7 DOUBLE PRECISION	,
  bet_13 DOUBLE PRECISION	,
  bet_14 DOUBLE PRECISION	,
  buy_flag_3 DOUBLE PRECISION	,
  buy_flag_4 DOUBLE PRECISION	,
  buy_flag_5 DOUBLE PRECISION	,
  buy_flag_7 DOUBLE PRECISION	,
  buy_flag_13 DOUBLE PRECISION	,
  buy_flag_14 DOUBLE PRECISION	,
  total_use INTEGER
);

/*bet,getそれぞれの合計のテーブル，集計用（日付単位でユニーク）*/
CREATE TABLE proba_test.bet_get_log_former_asiya_proba_test_2022_th09(
    place_name VARCHAR ,
    date text ,
    money INTEGER,
    money_type VARCHAR

);

/*get時の結果をまとめておく（自分の要素入っておらず，結果だけが格納されている），bet時と同様にレース単位でユニーク*/
CREATE TABLE proba_test.result_log_former_asiya_proba_test_2022_th09(
    place_name VARCHAR ,
    date text,
    num_race INTEGER ,
    result_coms INTEGER ,
    return_money INTEGER
);

/*ミスった時用のテーブル削除用のクエリ*/
DROP TABLE proba_test.bet_log_former_asiya_proba_test_2022_th09;
DROP TABLE proba_test.bet_get_log_former_asiya_proba_test_2022_th09;
DROP TABLE proba_test.result_log_former_asiya_proba_test_2022_th09;





























/*ミスった時用のテーブル削除用のクエリ*/
DROP TABLE proba_test.bet_log_former_asiya_proba_test_2022;
DROP TABLE proba_test.bet_get_log_former_asiya_proba_test_2022;
DROP TABLE proba_test.result_log_former_asiya_proba_test_2022;


















CREATE TABLE proba_test.bet_log_former_asiya_proba_test_2021(
    place_name VARCHAR ,
    date text,
    num_race INTEGER ,
    racer_1_id INTEGER,
    racer_2_id INTEGER,
    racer_3_id INTEGER,
    racer_4_id INTEGER,
    racer_5_id INTEGER,
    racer_6_id INTEGER,
    pred_3 DOUBLE PRECISION	,
    pred_4 DOUBLE PRECISION	,
    pred_5 DOUBLE PRECISION	,
    pred_7 DOUBLE PRECISION	,
    pred_13 DOUBLE PRECISION	,
    pred_14 DOUBLE PRECISION	,
    bet_3 DOUBLE PRECISION	,
    bet_4 DOUBLE PRECISION	,
    bet_5 DOUBLE PRECISION	,
    bet_7 DOUBLE PRECISION	,
    bet_13 DOUBLE PRECISION	,
    bet_14 DOUBLE PRECISION	,
    buy_flag_3 DOUBLE PRECISION	,
    buy_flag_4 DOUBLE PRECISION	,
    buy_flag_5 DOUBLE PRECISION	,
    buy_flag_7 DOUBLE PRECISION	,
    buy_flag_13 DOUBLE PRECISION	,
    buy_flag_14 DOUBLE PRECISION	,
    total_use INTEGER
);

/*bet,getそれぞれの合計のテーブル，集計用（日付単位でユニーク）*/
CREATE TABLE proba_test.bet_get_log_former_asiya_proba_test_2021(
    place_name VARCHAR ,
    date text ,
    money INTEGER,
    money_type VARCHAR

);

/*get時の結果をまとめておく（自分の要素入っておらず，結果だけが格納されている），bet時と同様にレース単位でユニーク*/
CREATE TABLE proba_test.result_log_former_asiya_proba_test_2021(
    place_name VARCHAR ,
    date text,
    num_race INTEGER ,
    result_coms INTEGER ,
    return_money INTEGER
);


DROP TABLE proba_test.bet_log_former_asiya_proba_test_2021;
DROP TABLE proba_test.bet_get_log_former_asiya_proba_test_2021;
DROP TABLE proba_test.result_log_former_asiya_proba_test_2021;
