# 不均衡データの実務的対応：評価指標と閾値調整

## 概要

ML教科書では「クラスバランスをとりましょう」と書かれています。

が、実務での分類問題は、その多くが不均衡データです。
本ドキュメントは、複数の金融データセット・マーケティング施策を扱った経験から、
不均衡データへの実務的対応戦略を記載します。

---

## 現場で直面する現実

### パターン1：正解データが極度に少ない

```
全データ：100,000件
正解（ポジティブ）：342件  ← 0.34%
不正解（ネガティブ）：99,658件

精度 99.66% で出したモデル
→ すべてネガティブと予測しただけ...
```

### パターン2：ビジネス的に見逃しが致命的

```
詐欺検知
- 正解：詐欺取引（0.1%）← 見逃しが高コスト
- 不正解：正常取引（99.9%）

マーケティング施策応答予測
- 正解：キャンペーン応答（5%）← 接触コストがある
- 不正解：応答しない（95%）
```

### パターン3：正解特徴に「幅」がある

```
マーケティング施策応答予測
- 明確に応答：Yes（クリック、購買）
- 応答しない：No
  - 全く興味ない（確定的）
  - 後で購買する可能性がある（不明確）
  - 見ていない（不明確）
```

---

## ビジネス要件による対応の分類

### タイプ A：見逃しが高コスト（コンプライアンス・リスク）

```
例）
- 詐欺検知：1件の見逃し = 数十万円の損失
- 医療診断：1件の見逃し = 患者の生命喪失
- セキュリティ脅威検知：1件の見逃し = 深刻なインシデント

対応戦略：Recall 重視
```

### タイプ B：誤検知が高コスト（接触コスト・ユーザー体験）

```
例）
- マーケティング施策：無関心な顧客に過剰接触 = ブランド毀損
- スパム検知：正常メール削除 = ユーザー不信
- 疾病リスク予測：無用な治療推奨 = 医療費無駄・副作用

対応戦略：Precision 重視
```

### タイプ C：見逃しと誤検知のバランスが必須

```
例）
- 一般的な分類タスク
- 両方のコストを考慮する必要がある

対応戦略：F1 or AUROC 重視
```

---

## 評価指標の選択

### 悪い例：不均衡データでAccuracy偏重

```python
accuracy = (TP + TN) / (TP + TN + FP + FN)

# 不均衡データの場合
# TN（ネガティブ正解）の数が圧倒的に多いため
# すべてネガティブと予測してもaccuracyが高い
# → 意味ある指標ではない
```

### 良い例：ビジネス要件別の指標選択

#### 1. Recall 重視（見逃しが致命的）

```python
Recall = TP / (TP + FN)
= 実際のポジティブの中で、正しく検知できた割合

例）詐欺検知で Recall = 95% の意味：
実際の詐欺 100件のうち、95件は検知された
5件は見逃された
```

#### 2. Precision 重視（誤検知が致命的）

```python
Precision = TP / (TP + FP)
= 検知したポジティブの中で、実際にポジティブだった割合

例）マーケティング施策で Precision = 80% の意味：
施策対象に選んだ 100人のうち、80人は実際に応答
20人は応答しなかった
```

#### 3. F1スコア（バランス型）

```python
F1 = 2 * (Precision * Recall) / (Precision + Recall)
= Precision と Recall の調和平均

Precision と Recall の両方を考慮したい場合
```

#### 4. PR曲線 と ROC曲線

```python
# 不均衡データでは ROC曲線より PR曲線が推奨
# (PR曲線は positive class に敏感)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
```

---

## 実装戦略

### Step 1：クラスウェイト調整

```python
from sklearn.linear_model import LogisticRegression

# クラス不均衡を学習時に考慮
model = LogisticRegression(
    class_weight='balanced',  # ← 少数派に重みをつける
    random_state=42
)

# または
model = LogisticRegression(
    class_weight={0: 1, 1: 10},  # ← ポジティブに10倍の重み
    random_state=42
)
```

### Step 2：閾値調整（極めて重要）

```
デフォルト閾値：0.5

分類ルール：
- pred_proba >= threshold → ポジティブと予測
- pred_proba < threshold → ネガティブと予測
```

#### 2-1. Recall 重視の場合

```python
# 閾値を低く設定（例：0.3）
# より多くのサンプルを「ポジティブ」と予測

threshold = 0.3
y_pred = (y_pred_proba >= threshold).astype(int)

# 結果
# - TP：増加（良い）
# - FN：減少（良い）
# - FP：増加（悪い） ← 許容範囲内
# → Recall は高くなるが Precision は低下
```

#### 2-2. Precision 重視の場合

```python
# 閾値を高く設定（例：0.7）
# より確度の高いだけ「ポジティブ」と予測

threshold = 0.7
y_pred = (y_pred_proba >= threshold).astype(int)

# 結果
# - TP：可能な限り正しい
# - FP：減少（良い）
# - FN：増加（悪い） ← 許容範囲内
# → Precision は高くなるが Recall は低下
```

### Step 3：可視化による確認

```python
import matplotlib.pyplot as plt
import numpy as np

# 予測確率分布を可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ネガティブクラスの予測確率分布
axes[0].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Negative (actual)')
# ポジティブクラスの予測確率分布
axes[0].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Positive (actual)')
axes[0].axvline(x=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].set_title('Prediction Probability Distribution')

# 混同行列
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center')
plt.newticks_locator()
plt.show()
```

---

## 実務での具体例

### 例1：詐欺検知（Recall 重視）

```
ビジネス要件：
- 詐欺見逃しはゼロに近づけたい
- 目標：Recall >= 95%

実装：
- 閾値を 0.2 に設定
- class_weight = {0: 1, 1: 50}

結果：
- Recall = 96%（詐欺 100件中 96件検知）
- Precision = 45%（検知 200件中 96件が実詐欺）
- → 200件の「疑わしいケース」を人間が精査
- → 96件の詐欺を同時に阻止

運用上の判断：
- 200件の精査コストと 96件の詐欺阻止のバランス
- ROI的に成り立つか確認
```

### 例2：マーケティング施策応答予測（Precision 重視）

```
ビジネス要件：
- キャンペーン接触コストが高い
- 無関心層への接触は避けたい
- 目標：Precision >= 70%

実装：
- 閾値を 0.7 に設定
- class_weight = {0: 1, 1: 2}

結果：
- Precision = 72%（対象 1000人中 720人応答）
- Recall = 35%（全応答 2000人中 720人を対象化）
- → 効率的で高品質な施策を実行
- → 見逃した 1280人は次回以降対象化

運用上の判断：
- 接触コスト削減と顧客体験を最適化
```

### 例3：バランス型（金融リスク評価）

```
ビジネス要件：
- 見逃しも誤検知も許容できない
- 定期的な人間レビューが入る
- 目標：F1 >= 0.70

実装：
- 閾値を 0.5（デフォルト）
- class_weight = 'balanced'

結果：
- Precision = 68%
- Recall = 73%
- F1 = 0.70
- → バランスの取れた運用
```

---

## よくある誤り

### 誤り1：精度偏重

```python
# ❌ 悪い例
best_model = max(models, key=lambda m: accuracy_score(y_test, m.predict(X_test)))

# ✅ 良い例
best_model = max(models, key=lambda m: f1_score(y_test, m.predict(X_test)) 
                 if business_requirement == 'balance' 
                 else recall_score(y_test, m.predict(X_test)))
```

### 誤り2：閾値調整なしの運用

```python
# ❌ 悪い例
model.predict(X) → 0/1 直接出力

# ✅ 良い例
y_pred_proba = model.predict_proba(X)[:, 1]
y_pred = (y_pred_proba >= optimal_threshold).astype(int)
```

### 誤り3：ビジネス要件を無視した評価

```python
# ❌ 悪い例
「AUCが0.85なので本番導入」

# ✅ 良い例
「Recall=92%（詐欺見逃し8%）で、
 誤検知が200件（Precision=60%）。
 運用コストと照合して採用判断」
```


## References

- Class Imbalance: https://scikit-learn.org/stable/modules/ensemble.html#class-weight
- Precision-Recall Trade-off: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
- F1 Score: https://en.wikipedia.org/wiki/F-score

