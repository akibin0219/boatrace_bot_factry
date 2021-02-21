module/   :便利なモジュールが入ったディレクトリ
--graph.py : グラフ化や分析に便利な関数のまとめ。
--master.py : 会場の番号と名前（ディレクトリ名）を持ったディクショナリを返す関数を持つ。
--trans_text_code.py : 全会場のテキストデータの文字コードの変換用の関数を持ったモジュール

analysis/  :pythonで基本的に行う分析のスクリプトのまとめフォルダ(モデルの分析など)
-asiya_model_analysis.ipynb:現状(2021/02月時点)で最もうまくいっている芦屋のモデルが何を考えて判別をしているのかを分析する。

modeling/  :モデリングを行うスクリプトのまとめフォルダ、基本的にバージョンで管理。(databaseの各会場のscore内にスコアを出力、大体一日半くらいで探索は終わる)
-modeling_{vertsion}.ipynb   バージョンごとの機能に関しては別途フォルダ内のREADMEに記述



HTML_data.ipynb:全会場の競争データをクローリングで取得し出力(.txt)

trans_text_code.ipynb:HTMLの文字コードを変換するスクリプト

making_traindata.ipynb:HTMLから情報を抜き出してCSVに保存、さらにパラメータの情報を結合してtrainのデータとして出力（databaseに出力）

model_analysis.ipynb:XGboost の予測を閾値で区切ったものの中で成績がいいモデルを探し、pickleで保存、加えてもでるが正常かどうかチェックするcsvを出力するスクリプト