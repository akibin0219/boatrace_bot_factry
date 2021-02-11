import requests
from bs4 import BeautifulSoup
import time
import datetime
import os #ディレクトリ作成用
from tqdm import tqdm
#自作モジュールインポート
import module.master as master
import module.graph as graph
import sys
import codecs

def result_text_trans():
    place_master=master.get_place_master()
    for place in place_master.items():
        place_name=place[1]
        dir_path = "../bot_database/{place_name}/{place_name}_result_txt_utf8/".format(place_name=place_name)
        if os.path.exists(dir_path)==False:
            os.makedirs(dir_path)
        else:
            pass
    #テキストファイルの文字コードの変換==========================================================
    #for place in tqdm(place_master_2.items()):
    for place in tqdm(place_master.items()):
        place_name=place[1]
        url_list=[]
        years=['2012','2013','2014','2015','2016','2017','2018','2019','2020']
        for year in years:
            read_fpath="../bot_database/{place_name}/{place_name}_result_txt/{year}_result_{place_name}.txt".format(place_name=place_name,year=year)
            write_fpath="../bot_database/{place_name}/{place_name}_result_txt_utf8/{year}_result_{place_name}_utf8.txt".format(place_name=place_name,year=year)
            try:
                sf = codecs.open(read_fpath, 'r', encoding='shift-jis')
                uf = codecs.open(write_fpath, 'w', encoding='utf-8')
                for line in sf:
                    uf.write(line)
                sf.close()
                uf.close()
            except FileNotFoundError:
                print('file_not_found_error!!!!!!!!!!',place_name,year)


            """
            with open(read_fpath, encoding='ANSI') as f_in:
                with open(write_fpath, 'w', encoding='utf8') as f_out:
                    f_out.write(f_in.read())
            """
    return None
