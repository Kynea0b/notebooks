# 追加学習モデル

datasetで追加学習したモデルのGoogle Drive上のidと追加学習に使用したデータセットの説明。
追加学習済みのモデルを使用して推論を実行する用の処理。

## experimental_001

`sentiment_analysis_huggingface+wrime`

dataset: [wrime](https://github.com/ids-cv/wrime)

```bash
! wget https://github.com/ids-cv/wrime/raw/master/wrime-ver1.tsv
```

## 実験メモ；model比較

感情分類ではほぼ差がない。
bert-base-japanese-v3
bert-base-japanese-whole-word-masking
