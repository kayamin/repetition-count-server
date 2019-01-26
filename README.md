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
