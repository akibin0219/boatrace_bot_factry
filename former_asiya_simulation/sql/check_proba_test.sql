SELECT
    *
FROM
    proba_test.bet_log_former_asiya_proba_test_2022
;



SELECT
    *
FROM
    proba_test.bet_get_log_former_asiya_proba_test_2022
;



SELECT
    *
FROM
    proba_test.result_log_former_asiya_proba_test_2022
;





/*会場ごとに投資金額と収益をすべて出す。*/
WITH get AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS get_money
  FROM
     proba_test.bet_get_log_former_asiya_proba_test_2022 l
     --proba_test.bet_get_log_former_asiya_proba_test_2022_th07 l
     --proba_test.bet_get_log_former_asiya_proba_test_2022_th09 l
    -- proba_test.bet_get_log_former_asiya_proba_test_2021 l

  WHERE
      money_type='get'
), bet AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS bet_money
  FROM
     proba_test.bet_get_log_former_asiya_proba_test_2022 l
     --proba_test.bet_get_log_former_asiya_proba_test_2022_th07 l
     --proba_test.bet_get_log_former_asiya_proba_test_2022_th09 l
     --proba_test.bet_get_log_former_asiya_proba_test_2021 l

 WHERE
     money_type='bet'
), join_g_b AS(
  SELECT
      g.date
      , g.place_name
      , get_money
      , bet_money
  FROM get g
  LEFT JOIN bet b
      ON g.date=b.date
      AND g.place_name=b.place_name
)
SELECT
    place_name
    , SUM(get_money) AS get
    , SUM(bet_money) AS bet
FROM join_g_b
GROUP BY
    place_name
;









/*会場ごとに投資金額と収益をすべて出す。*/
WITH get AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS get_money
  FROM
  proba_test.bet_get_log_former_asiya_proba_test_2022 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th07 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th09 l
  --proba_test.bet_get_log_former_asiya_proba_test_2021 l

  WHERE
      money_type='get'
      AND cast(l.date as integer) >=20210101
      AND cast(l.date as integer) <20220101
), bet AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS bet_money
  FROM
  proba_test.bet_get_log_former_asiya_proba_test_2022 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th07 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th09 l
  --proba_test.bet_get_log_former_asiya_proba_test_2021 l
 WHERE
     money_type='bet'
     AND cast(l.date as integer) >=20210101
     AND cast(l.date as integer) <20220101
), join_g_b AS(
  SELECT
      g.date
      , g.place_name
      , get_money
      , bet_money
  FROM get g
  LEFT JOIN bet b
      ON g.date=b.date
      AND g.place_name=b.place_name
)
SELECT
    place_name
    , SUM(get_money) AS get
    , SUM(bet_money) AS bet
FROM join_g_b
GROUP BY
    place_name
;





/*月ごとの収益計算*/

WITH get AS(
  SELECT
      CAST(l.date as date ) AS trans_date
      , extract(YEAR from CAST(l.date as date )) AS year
      , extract(MONTH from CAST(l.date as date )) AS month
      , l.place_name
      , l.money AS get_money
  FROM
    proba_test.bet_get_log_former_asiya_proba_test_2022 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th07 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th09 l
  --proba_test.bet_get_log_former_asiya_proba_test_2021 l

  WHERE
      money_type='get'
      -- AND cast(l.date as integer) >=20210101
      -- AND cast(l.date as integer) <20220101
), bet AS(
  SELECT
      CAST(l.date as date ) AS trans_date
      , extract(YEAR from CAST(l.date as date )) AS year
      , extract(MONTH from CAST(l.date as date )) AS month
      , l.place_name
      , l.money AS bet_money
  FROM
    proba_test.bet_get_log_former_asiya_proba_test_2022 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th07 l
  --proba_test.bet_get_log_former_asiya_proba_test_2022_th09 l
  --proba_test.bet_get_log_former_asiya_proba_test_2021 l
 WHERE
     money_type='bet'
     -- AND cast(l.date as integer) >=20210101
     -- AND cast(l.date as integer) <20220101
), join_g_b AS(
  SELECT
      g.trans_date
      , g.month
      , g.year
      , g.place_name
      , get_money
      , bet_money
  FROM get g
  LEFT JOIN bet b
      ON g.trans_date=b.trans_date
      AND g.month=b.month
      AND g.place_name=b.place_name
)
SELECT
    place_name
    , month
    , SUM(get_money) AS get
    , SUM(bet_money) AS bet
FROM join_g_b
GROUP BY
    place_name
    , year
    , month
ORDER BY
    year
    , month
;
