/*行った投票のデータをDBからとってくる用のSQL(収益計算時の関数に埋め込まれているやつ)*/
SELECT
*
FROM
    proba_test.bet_log_former_asiya_proba_test_2022 b
    --proba_test.bet_log_former_asiya_proba_test_2021 b
    --proba_test.bet_log_former_asiya_proba_check_test_2022 b
    --proba_test.bet_get_log_former_asiya_proba_check_test_2022 b
    --proba_test.result_log_former_asiya_proba_check_test_2022 b
-- WHERE
--     b.date=cast({target_date} AS text)
--     AND b.place_name='{place_name}'
;
