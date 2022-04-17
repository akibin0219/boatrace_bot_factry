scraping_crawling:スクレイピングとクローリングで配当、選手のID、着順をネットからとってくる
making_traindata :スクレイピングしたCSVとネットでダウンロードできるパラメータCSVとくっつけて学習データを作る。
---[making_traindata内] asiya_result_csv     : スクレイピングしたデータを入れておく
　                      racerpara_by_year_csv:各年の選手のパラメータを置いておく場所
                        train_by_year_asiya  :　トレインデータを一年ごとに出力するためのフォルダ（需要はほとんどない）
　　　　　　　　　　　　join_toll.ipynb      :racer_IDとパラメータをくっつけて学習データを出力するノートブック

shulkei          :データの基礎集計用のフォルダ(平均値、中央値での期待値、分布など)
score_analysis   :探索的に探したcom_パラメータを評価


