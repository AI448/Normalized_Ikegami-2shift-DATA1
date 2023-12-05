# このリポジトリに含まれるファイルについて

## Ikegami-2shift-DATA1.xlsx

池上先生が公開されているデータ[3]を正規化したものになります．

* 元のデータでは ```--```, ```/```, ```N```, ```n```, ```+``` となっていたシフトの種別を，それぞれ ```DAY```, ```OFF```, ```NIGHT```, ```MORNING```, ```OTHER``` に変更しています．
* 元のデータでは日付が 1, ..., 31 と具体的な日付ではありませんでしたが，このデータでは 2023/9/1, ..., 2023/9/31 に変更しています．

## sample.py

上記データを読み込んで，最適化を実行するサンプルスクリプトになります．
データ[3]と参考文献[2]から意味を解釈して定式化していますが，前月の勤務の考慮を省略しています．

```sh
python sample.py Ikegami-2shift-DATA1.xlsx
```

# 注意事項

データ変換時のミス・モデルの解釈間違い等が存在する可能性があるため，論文の結果を再現する目的で使用される場合には，ご自身で確認・修正をお願いいたします．

# 参考文献

* [1] Ikegami, A., Niwa, A. (2003): "A Subproblem-centric Model and Approach to the Nurse Scheduling Problem", Mathematical Programming 97, 517-541.
* [2] [池上敦子, 丹羽明, 大倉元宏(1996): "我が国におけるナース・スケジューリング問題", オペレー
ションズ・リサーチ, Vol. 41, No. 8, pp. 436-442.](https://orsj.org/wp-content/or-archives50/pdf/bul/Vol.41_08_436.pdf)
* [3] http://cricket.ikegami-lab.tokyo/~atsuko/DATA/data.html