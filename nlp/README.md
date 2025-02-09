# NLPタスク

## sentiment_analysis_huggingface+wrime

日本語の感情分類を行うモデル。

日本語感情分析に使用するモデルたち。
https://github.com/cl-tohoku/bert-japanese/

追加学習するデータセット。
https://github.com/ids-cv/wrime

## background

[BERT](https://qiita.com/omiita/items/72998858efc19a368e50)




## detail

### モデル別にNLPタスクを実行させてみよう

- "cl-tohoku/bert-base-japanese-whole-word-masking"
- "https://huggingface.co/tohoku-nlp/bert-base-japanese-v3"

```python
# 使用するモデルを指定して、Tokenizerを読み込む
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
```

## モデルの変更

>機械学習モデルに入力するデータの形状は、モデルが期待する形式に合わせる必要があります。特に、自然言語処理（NLP）の分野では、テキストデータをトークン化し、特定の形式（例えば、系列長、バッチサイズなど）に変換する必要があります。

モデル名与えただけではうまくいかない
モデルごとに入力データの形状が決められている。

```error
expected sequence of length 19 at dim 1: モデルは、次元1（通常は系列長）が19のシーケンス（系列データ）を期待しています。
got 10: 実際には、次元1が10のシーケンスが入力されました。
つまり、モデルが19個の要素を持つ系列データを期待しているのに対し、実際には10個の要素を持つ系列データが与えられたため、エラーが発生したということです
```

はい、おっしゃる通りです。モデルごとに入力データの形状（shape）は基本的に決まっています。

入力データの形状とは
機械学習モデルに入力するデータの形状は、モデルが期待する形式に合わせる必要があります。特に、自然言語処理（NLP）の分野では、テキストデータをトークン化し、特定の形式（例えば、系列長、バッチサイズなど）に変換する必要があります。

### モデルごとの違い

系列長: モデルが扱うことができるテキストの最大長（系列長）は、モデルによって異なります。BERT系のモデルでは、通常、512トークンや1024トークンが上限となっています。
バッチサイズ: 複数のデータをまとめて処理する場合（バッチ処理）、バッチサイズを指定する必要があります。バッチサイズは、モデルのメモリ容量や計算資源によって制限されます。
入力形式: モデルによっては、入力データの形式（例えば、トークンの種類、特殊トークンの有無など）が異なる場合があります

## ipynbを.pyに変換

```python
pip install nbconvert
```

```python
jupyter nbconvert --to python [ipynbファイル名]
```

## 学習モデルの保存

GPU実行回数の節約のために、学習モデルを追加学習したら、追加学習済みモデルは保存する。
https://qiita.com/ryu_pro1000/items/c28c1307a4958e4e31ac
