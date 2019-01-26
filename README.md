# repetition-count-server

# Live Repetition Counting のモデルのiOS アプリに実装のサーバーサイド実装

```
@InProceedings{Levy_2015_ICCV,
author = {Levy, Ofir and Wolf, Lior},
title = {Live Repetition Counting},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}
```

repetition-count-server
- repetiion-count-app から送られてくる予測結果を元に，カウント値を算出する api, 現在のカウント値を返す api を実装したサーバー

repetition-count-app（別レポジトリ）
- カメラからの動画取り込み
- OpenCV を用いてROI抽出
- CoreMLで実装した学習済み深層学習モデルを用いて， 取り込んだ20frame 分の画像に対して cycle length を予測
- repetition-count-server に 予測結果を送り， 現在のカウント数を取得


著者による論文実装との違い : https://github.com/tomrunia/DeepRepICCV2015
- theano による深層学習モデルと, numpy等を用いたカウントロジックをそれぞれ，iOSアプリ(CoreML)， サーバーに分離



# 実行手順
## 1. mlmodel の作成
1. 著者実装の theanoモデルを keras モデルに変換する
  - theano -> pytorch : 重みの reshape のみで可能
  - pytorch -> kears : pytorch2keras を利用して変換
  
2. coremltools を用いて kereas モデルを CoreMLで用いる mlmodel へ変換

## 2. サーバの起動
1. repository のルートディレクトリで下記コマンドを実行してサーバーを起動
  - 起動する IPアドレスを 0.0.0.0 とすることで， iOSアプリから接続可能
  - アプリを起動する iPhone と サーバーを起動する PC は同じネットワーク接続している必要がある
```
python manage.py runserver 0.0.0.0:8000
```

## 3. iOSアプリ(repetition-count-app)の起動，カウント
1. 1. で作成した mlmodel を repetition-count-app のルートディレクトリにコピー
2. iPhone を USBケーブルで PCに接続し， iPhone を対象に run
3. 回数カウントしたい動作を iPhone で写す
  - 最初に reset を押すことでサーバーとの通信をセットアップ

