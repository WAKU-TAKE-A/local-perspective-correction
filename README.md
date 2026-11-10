# local-perspective-correction

画像上の局所分解能を用いた平面透視補正

画像上の複数地点で得た方向付き局所分解能（mm/pixel）から、平面の
ホモグラフィを推定し、透視歪みを補正するシステムです。
設計と数学モデルは [specification.md](specification.md)（v0.5）を参照してください。

## 使い方

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt   # Windows では venv\Scripts\pip
venv/bin/python app.py
```

GUI は左側が制御系、右側がログとエラー表示です。

- 角度 θ は利用者が決めるパラメータです（スライダー、初期値 90°）。
  推定結果は θ に対して返ります。自動提案はしません。
- 測定データは表に直接入力するか、CSV で読み込めます。
  CSV の必須列は `x,y,period,element,group` です
  （period は画像上の周期 pixel、局所分解能は ピッチ / period で計算）。
  `dirx,diry`（測定方向）は任意で、空欄の場合は同じ要素の測定点を通る
  直線から自動計算します（要素に1点しかない場合は、同方向群の消失点から
  求めます）。
- 「合成サンプル生成」で、検証用の合成データ（真値つき）を試せます。

## テスト

```bash
venv/bin/python -m pytest tests/
```

## 構成

- `lpc/core.py` — ホモグラフィ推定（中心制約・平行性・θ パラメータ、入力不足の判定、ランクと条件数の評価）、画像のワープ
- `lpc/simulate.py` — ピンホールカメラモデルによる検証用データ生成
- `lpc/gui.py` — ttkbootstrap 製 GUI
- `app.py` — 起動エントリ
