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
  WHERE
      money_type='get'
), bet AS(
  SELECT
      l.date
      , l.place_name
      , l.money AS bet_money
  FROM
     proba_test.bet_get_log_former_asiya_proba_test_2022 l
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
