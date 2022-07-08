#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import requests
import gzip
import time


def main():
    # APIのurl
    url = "http://api.syosetu.com/novelapi/api/"

    # ジャンル指定
    genres = ["201", "303", "401", "402", "404"]

    # 条件を満たす小説のうち1, 501, 1001, 1501, 2000件目に続く500件のデータをダウンロードする。
    # ただし、仕様上選択できる開始地点の最大値は2000であるため、
    # 2000件目のデータが重複することを防ぐために
    # 1501件目から始まる区間は499件ダウンロードする。
    starts = [1, 501, 1001, 1501, 2000]

    data = []

    for genre in genres:
        for start in starts:
            lim = 500 if start != 1501 else 499

            # クエリパラメータによって出力設定を渡す
            payload = {
                # ncode(小説に固有のid), タイトル, あらすじ, ジャンルを出力
                "of": "n-t-s-g",
                # 総合ポイント順に出力
                "order": "hyoka",
                # 出力形式はjson
                "out": "json",
                # gzip形式、圧縮率最大
                "gzip": "5",
                # ジャンル指定
                "genre": genre,
                # 開始地点指定
                "st": start,
                # 出力件数指定
                "lim": lim,
            }

            # リクエスト
            response = requests.get(url, params=payload)

            # サーバ負担軽減のためgzip形式でダウンロード
            response.encoding = "gzip"

            # デコード
            r = response.content
            res_content = gzip.decompress(r).decode("utf-8")
            response_json = json.loads(res_content)

            # print(response_json)

            # ダウンロードしたデータの先頭以外をリストに格納する。
            # 先頭には条件を満たす小説の総数が入っている。
            data.extend(response_json[1:])

            # サーバ負担軽減のため30秒待機
            time.sleep(30)

    # json形式のファイルに出力
    with open("./data.json", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
