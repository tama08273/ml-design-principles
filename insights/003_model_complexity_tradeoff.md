# モデル複雑性と選択基準：ロジスティック回帰から勾配ブーストまで

## 概要

ML初心者は「複雑なモデル = 精度が高い」と考えがちです。

が、実務での複数プロジェクト経験から、
データ複雑性、解釈性要件、運用コストに応じた**最適なモデル選択**が
精度向上より重要であることを学びました。

本ドキュメントは、ロジスティック回帰→決定森→勾配ブーストの
使い分けと、複雑性-精度-解釈性のトレードオフについて記載します。

---

## モデルの系統と複雑性

### 図解：複雑性スペクトラム

```
解釈性
高  ↑
    │ ◆ ロジスティック回帰
    │   + 決定木（Shallow）
    │ ◆ 決定森（Random Forest）
    │   ◆ 勾配ブースト（XGBoost, LightGBM）
    │       ◆ ニューラルネット（FeedForward）
    │           ◆ Transformer, LSTM
    └───────────────────────→
低   複雑性（計算量、ハイパーパラメータ数、ブラックボックス度）
```

---

## 現場で直面する現実

### シナリオ1：金融機関の与信審査

```
要件：
- 申込者Aに「融資判定：否決」と伝える際、
  「なぜ否決か」を説明する必要がある
  （金融規制要件）

ロジスティック回帰の場合：
  y = 0.05 * 年齢 + 0.2 * 年収 - 0.8 * 負債比率 + ...
  → 「年収と負債比率が主な否定要因」と説明可能

複雑なブーストモデルの場合：
  → 「複雑な相互作用で判定しました」と説明不可能
  → 規制に違反
```

### シナリオ2：ヘルスケアアプリの疾病リスク予測

```
要件：
- 予測精度 95%
- ただし「この値を処方判定に直接使用しない」
  （医師が最終判断）

判定ルール：
- Risk > 0.9 → 医師へモニタリング推奨アラート
- 0.5 < Risk < 0.9 → 医師へ参考情報
- Risk < 0.5 → アラートなし

選択：ロジスティック回帰または浅い決定木
理由：
- 医師が理由を理解→信頼
- 複雑モデルで精度が 96% でも、医師が理由を理解できず
  → 使われない
- 結果：ロジスティック回帰（精度90%）のほうが実用的
```

### シナリオ3：マーケティング施策での応答予測

```
要件：
- 予測精度が高い
- 説明性は重要ではない（施策対象決定に使うだけ）
- 性能競争が激しい（1% 精度向上で ROI 向上）

選択：勾配ブースト（XGBoost, LightGBM）
理由：
- 説明性不要 → 複雑モデル OK
- 精度競争 → 複雑モデルの性能差が重要
- 本番環境が安定 → 過学習リスク許容
```

---

## モデル選択の判定基準

### ステップ1：要件整理

```
□ 1. 解釈性は必須か？
     YES → ロジスティック回帰・浅い決定木
     NO → 複雑モデル検討

□ 2. トレーニングデータサイズ
     < 1万行 → シンプルモデル（過学習回避）
     1万～100万行 → 中程度モデル
     > 100万行 → 複雑モデル可

□ 3. 特徴量数
     < 50 → シンプルモデル
     50～1000 → 中程度モデル
     > 1000 → 複雑モデル（正則化必須）

□ 4. 本番環境安定性
     不安定 → シンプルモデル（過学習リスク低）
     安定 → 複雑モデル
```

### ステップ2：データセット分析

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 各モデルの交差検証スコア計測
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"{name}: Mean F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 各モデルの特性

### パターンA：ロジスティック回帰

```
用途：
- 金融（与信、詐欺検知）
- ビジネス指標の要因分析
- リアルタイム推論が必要
- 規制対応が厳しい

利点：
+ 解釈性が最高
+ 計算が高速
+ ハイパーパラメータが少ない
+ 理論的基盤が強い

欠点：
- 非線形関係が捉えられない
- 特徴量エンジニアリングが重要
- 精度は限定的

実装例：
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 特徴量正規化は必須
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(
    C=1.0,  # 正則化強度の逆数（小さいほど強い）
    solver='lbfgs',  # または 'liblinear'
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

# 係数を解釈
import pandas as pd
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

print(coef_df)
# → 各特徴量の影響度を直接理解可能
```

### パターンB：決定木（Random Forest）

```
用途：
- 中程度の解釈性で高精度が必要
- 非線形関係が複雑
- 外れ値に強いモデルが必要

利点：
+ 非線形関係を捉える
+ 特徴量エンジニアリング最小限
+ 外れ値に強い
+ 並列処理可能
+ Feature Importance が解釈可能

欠点：
- ロジスティック回帰より計算量が多い
- ハイパーパラメータが多い
- 解釈性は低下

実装例：
```

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # 浅めに（過学習回避）
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 特徴量の重要度
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Features by Random Forest')
plt.show()
```

### パターンC：勾配ブースティング決定木（XGBoost / LightGBM）

```
用途：
- 最高精度が求められる
- データサイズが大きい
- 解釈性は二次的

利点：
+ 最高クラスの精度
+ 非線形関係を複雑に捉える
+ 大規模データ対応
+ SHAP値による事後解釈が可能
+ GPU対応で高速化可能

欠点：
- ハイパーパラメータ多い（チューニング大変）
- 過学習リスク高い
- 計算量が大きい（学習に時間）
- ブラックボックス度が高い

実装例：
```

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ハイパーパラメータグリッドサーチ
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# SHAP値による解釈
import shap

explainer = shap.TreeExplainer(grid_search.best_estimator_)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
# → ブラックボックスながら事後解釈が可能
```

---

## 複雑性-精度-解釈性のトレードオフ

### グラフ：一般的なトレンド

```
精度
100%│         ●（XGBoost）
    │       ●（Random Forest）
 85%│     ●（ロジスティック回帰）
    │   │
    │   └─── 解釈性が必須なら、ここまで
    │
 70%│
    └──────────────────────→
    シンプル  中程度  複雑
```

### 実務での判断例

#### 例1：金融リスク審査システム

```
要件優先度：
1. 解釈性（規制要件）
2. 精度（意思決定精度）
3. 速度（運用効率）

選択：ロジスティック回帰 + 手作り特徴量
精度損失：最大版との比較で 3-5%
→ 許容できる（解釈性による規制リスク回避が優先）
```

#### 例2：リアルタイム推奨エンジン

```
要件優先度：
1. 精度（ユーザー体験）
2. 速度（レイテンシ < 100ms）
3. 解釈性（ユーザーに「なぜ」は説明不要）

選択：LightGBM（GPU対応）
精度获得：3-5% 向上（ロジスティック回帰比）
→ ユーザー体験 → 採用率向上 → ROI 向上
```

#### 例3：ヘルスケア予測モデル

```
要件優先度：
1. 解釈性（医師の信頼）
2. 精度（医学的妥当性）
3. 速度（運用効率）

選択：浅い決定木 or ロジスティック回帰
精度確認：医師が理解できる範囲で許容精度確保
結果：精度 85% の説明可能モデル
    > 精度 92% のブラックボックスモデル
    （医師が使わない）
```

---

## ハイパーパラメータチューニングの実務的判断

### パターン1：シンプルモデルの場合

```
イテレーション2-3回で OK
- ロジスティック回帰：C値（正則化強度）の調整
- 決定木：max_depth を手動調整

効率的：学習時間が短く、過学習リスク低い
```

### パターン2：複雑モデルの場合

```
グリッドサーチ + 早期停止の両方使用

# 粗いグリッドサーチ
params_coarse = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3]
}
# → 大体の最適値発見

# 細かいグリッドサーチ
params_fine = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1, 0.15]
}
# → 最終調整

# 本番学習で早期停止
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10  # ← 過学習時点で学習停止
)
```

---

## よくある誤り

### 誤り1：「常に複雑なモデル = 最適」という誤解

```python
# ❌ 悪い実装
model = XGBClassifier(n_estimators=1000, max_depth=15)
model.fit(X_train, y_train)
# → 過学習の可能性高い
# → 本番環境で精度低下

# ✅ 良い実装
model = LogisticRegression()
model.fit(X_train_processed, y_train)
score_val = cross_val_score(model, X_val, y_val)
# → シンプルモデルでベースライン確立
# → 複雑モデルの追加は「必要に応じて」
```

### 誤り2：解釈性無視

```python
# ❌ 悪い実装
医療現場で最高精度のブーストモデル採用
→ 医師が「なぜ」を理解できず
→ 運用現場で使われず

# ✅ 良い実装
浅い決定木（精度 85%）で開始
→ 医師レビュー OK
→ 段階的に複雑化
→ SHAP解釈で信頼構築
```

### 誤り3：本番環境考慮なし

```python
# ❌ 悪い実装
開発環境でのF1スコア 0.95
本番環境でF1スコア 0.72 ← 過学習、データドリフト

# ✅ 良い実装
- 交差検証で安定性確認
- Holdout テストセット使用
- 本番環境のサブセットでテスト
- の過学習監視システム構築
```


## References

- Model Complexity: https://scikit-learn.org/stable/modules/model_selection.html
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://github.com/slundberg/shap
- Feature Engineering: https://en.wikipedia.org/wiki/Feature_engineering

