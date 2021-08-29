WITH trans_date AS(
  SELECT
      TO_DATE(l.date, 'YYYY/MM/DD') AS date --dateがたに変換
      , l.place_name
      , l.money
      , l.money_type
  FROM
     --former.former_bet_get_log_t_th05_all_hit2 l
     former.bet_get_log_former_all_v2_2 l
     --former.former_bet_get_log_t_th05_all l
     --former.bet_get_log_former_all_v2_2_2020 l
    -- former.former_bet_get_log_t_th05_all_hit2 l
     --former.former_bet_get_log_t_th05_all_hit2_2020 l

),get AS(
  SELECT
      extract(YEAR from l.date) AS year
      , extract(MONTH from l.date) AS month
      , l.place_name
      , SUM(l.money)AS get_money
  FROM
     trans_date l
  WHERE
      money_type='get'
  GROUP BY
      place_name
      , extract(YEAR from l.date)
      , extract(MONTH from l.date)
), bet AS(
  SELECT
      extract(YEAR from l.date) AS year
      , extract(MONTH from l.date) AS month
      , l.place_name
      ,SUM( l.money) AS bet_money
  FROM
     trans_date l
  WHERE
      money_type='bet'
  GROUP BY
      place_name
      , extract(YEAR from l.date)
      , extract(MONTH from l.date)
), join_g_b AS(
  SELECT
      b.place_name
      , b.year
      , b.month
      , get_money
      , bet_money
  FROM get g
  --LEFT JOIN bet b
  RIGHT OUTER JOIN bet b
      ON g.year=b.year
      AND g.place_name=b.place_name
      AND g.month=b.month
)
SELECT
    place_name
    , year
    , month
    , get_money
    , bet_money
    , get_money-bet_money AS income
FROM
    join_g_b
-- WHERE
--    month=5
--    OR month=4
--WHERE
--     (get_money-bet_money)>0
ORDER BY
    place_name
    , year
    , month

;


--開催日数に関する分析=========================================
WITH get_t AS(--ここで使用するテーブルを決定する
  SELECT
      *
  FROM
     date_former.{} l
 WHERE
     l.range_date>3 AND l.range_date<8
),get AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS get_money
      , l.year
      , l.month
      , l.day
      , l.num_date
      , l.range_date
      , l.season
  FROM
     get_t l
  WHERE
      money_type='get'
), bet AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS bet_money
      , l.year
      , l.month
      , l.day
      , l.num_date
      , l.range_date
      , l.season
  FROM
     get_t l
  WHERE
      money_type='bet'
), join_g_b AS(--情報の結合
  SELECT
      b.place_name
      , b.year
      , b.month
      , get_money
      , bet_money
      , get_money-bet_money AS income
      , b.day
      , b.num_date
      , b.range_date
      , b.season
  FROM get g
  RIGHT OUTER JOIN bet b
      ON g.year=b.year
      AND g.place_name=b.place_name
      AND g.month=b.month
      AND g.date=b.date
)
SELECT--場，開催日数ごとにまとめる
    j.place_name
    , j.range_date
    , SUM(j.income) AS income
    , COUNT(*) AS num_date
    , ROUND((CAST(COUNT(j.income>0 OR NULL) AS DEC)/COUNT(*))*100 , 3) AS num_plus_day
    , ROUND((CAST(COUNT(j.income<0 OR NULL) AS DEC)/COUNT(*))*100 , 3) AS num_minus_day

FROM
    join_g_b j
GROUP BY
    j.place_name
    , j.range_date
;




--何番目の開催日かに関する分析=========================================
WITH get_t AS(--ここで使用するテーブルを決定する
  SELECT
      *
  FROM
     date_former.{} l
 WHERE
     l.range_date>3 AND l.range_date<8
),get AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS get_money
      , l.year
      , l.month
      , l.day
      , l.num_date
      , l.range_date
      , l.season
  FROM
     get_t l
  WHERE
      money_type='get'
), bet AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS bet_money
      , l.year
      , l.month
      , l.day
      , l.num_date
      , l.range_date
      , l.season
  FROM
     get_t l
  WHERE
      money_type='bet'
), join_g_b AS(--情報の結合
  SELECT
      b.place_name
      , b.year
      , b.month
      , get_money
      , bet_money
      , get_money-bet_money AS income
      , b.day
      , b.num_date
      , b.range_date
      , b.season
  FROM get g
  RIGHT OUTER JOIN bet b
      ON g.year=b.year
      AND g.place_name=b.place_name
      AND g.month=b.month
      AND g.date=b.date
)
SELECT--場，開催日数ごとにまとめる
    j.place_name
    , j.num_date
    , SUM(j.income) AS income
    , COUNT(*) AS num_date
    , ROUND((CAST(COUNT(j.income>0 OR NULL) AS DEC)/COUNT(*))*100 , 3) AS num_plus_day
    , ROUND((CAST(COUNT(j.income<0 OR NULL) AS DEC)/COUNT(*))*100 , 3) AS num_minus_day

FROM
    join_g_b j
GROUP BY
    j.place_name
    , j.num_date
;











--====================================================
--テーブルの中身の確認
select
    COUNT(*)
from
    date_former.bet_get_log_former_all_v2_2
;
