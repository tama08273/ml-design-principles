# クロスバリデーションと楽観性バイアス：Nested CVによる真の汎化性能推定

## 概要

クロスバリデーションは「一般的なML実務の常識」です。

が、単純なクロスバリデーションだけでは、ハイパーパラメータチューニング時に
過学習が潜む可能性があります。本ドキュメントは、
**Nested Cross-Validation** による真の汎化性能推定と、
楽観性バイアス（Optimism Bias）の実装的対応を記載します。

---

## 現場で直面する現実

### シナリオ：モデル開発から本番導入

```
開発環境：
F1スコア = 0.92（交差検証）vs 本番環境：F1スコア = 0.71

「え、なぜ 20% 下がるの？」
→ 過学習 + データドリフト + 本番環境の特性
```

### 人間の直感的なバイアス

```
❌ 悪いアプローチ：
1. 訓練データで複数モデルを試す
2. 最も良い結果を選ぶ
3. テストセットで評価

何が起きるか？
→ 「最も良い結果」は、訓練データの特性に over-fit している
→ テストセットでの評価がオプティミスティック（楽観的）
```

---

## 楽観性バイアス（Optimism Bias）の理解

### 単純なバリデーション（危険）

```python
import numpy as np
from sklearn.model_selection import cross_val_score

# ハイパーパラメータグリッド
hyperparam_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]

best_score = -np.inf
best_hyperparam = None

for hp in hyperparam_candidates:
    model = LogisticRegression(C=hp)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    avg_score = scores.mean()
    
    if avg_score > best_score:
        best_score = avg_score
        best_hyperparam = hp

print(f"Best hyperparam: {best_hyperparam}")
print(f"CV score: {best_score}")

# 何が問題？
# → best_score は「訓練データ上での最高値」
# → ハイパーパラメータチューニングが訓練データに適応しすぎ
# → 未見データでの真の性能 < best_score
```

### 図解：楽観性バイアスの発生メカニズム

```
複数のハイパーパラメータ候補を試す
  ↓
各候補で交差検証スコア（訓練ベース）
  ↓
最も高いスコアの候補を選択
  ↓
選ばれたモデルは「訓練データに最適化」
  ↓
本番環境（未見データ）での性能は低下
```

### 数値例

```
ハイパーパラメータ候補：5 個
各候補の真の汎化性能：70%
ノイズ（ランダム変動）：±10%

ケース1：運が良い候補
  訓練跨CVスコア：75%（70% + 運の +5%）
  本番環境での真の性能：70%
  
運よく「75%」の候補を選ぶと...
  選別後の推定性能：75%
  実際の本番性能：70%
  → +5% の楽観性バイアス
```

---

## 解決策：Nested Cross-Validation

### 概念図

```
┌─────────────────────────────────────────┐
│          Outer Loop（5-fold CV）           │
│  性能評価 用の分割                    │
│                                     │
│  Train Set  │    Test Set          │
│  ┌────────────────┬─────────┐      │
│  │ Inner Loop (3×5-fold CV)│      │
│  │ ハイパーパラメータチューニング  │      │
│  │    （最適HPを選ぶ）       │      │
│  └────────────────┘      │      │
│  → 最適 HP で最終モデル   │      │
│  → Test Set で評価 →   Score │
│                                     │
└─────────────────────────────────────────┘

Outer Loop: 5回回転
各回で：
  - Inner Loop でハイパーパラメータ最適化
  - Outer Loop Test で最終評価
  - 5 つの独立した評価スコア → 平均と標準偏差
```

### 実装例

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Outer Loop: 5-fold CV（最終評価）
outer_cv = 5
scores = []

for train_idx, test_idx in KFold(n_splits=outer_cv).split(X):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner Loop: ハイパーパラメータ最適化
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
    
    inner_cv = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=3,  # Inner CV は 3-fold（計算量削減）
        scoring='f1'
    )
    
    inner_cv.fit(X_train_outer, y_train_outer)
    
    # Outer Loop テストセットで評価
    score = inner_cv.score(X_test_outer, y_test_outer)
    scores.append(score)
    
    print(f"Best C: {inner_cv.best_params_['C']:.4f}, Test Score: {score:.4f}")

# 最終評価
print(f"\nNested CV Average Score: {np.mean(scores):.4f}")
print(f"Nested CV Std Dev: {np.std(scores):.4f}")
```

### scikit-learn での便利なAPI

```python
from sklearn.model_selection import cross_validate

# より簡潔に書く場合
param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}

nested_cv_score = cross_validate(
    GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=3
    ),
    X, y,
    cv=5,  # Outer Loop は 5-fold
    scoring='f1'
)

print(f"Nested CV Score: {nested_cv_score['test_score']}")
print(f"Mean: {np.mean(nested_cv_score['test_score']):.4f}")
print(f"Std: {np.std(nested_cv_score['test_score']):.4f}")
```

---

## Nested CV vs 単純 CV の比較

### 実データでのシミュレーション

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression

# 訓練データ生成
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=10, random_state=42)

# ====== 単純 CV ======
print("Simple CV (BIASED):")
param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]}
best_score = -np.inf
best_param = None

for C in param_grid['C']:
    scores = cross_val_score(
        LogisticRegression(C=C, max_iter=1000),
        X, y, cv=5, scoring='f1'
    )
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_param = C

print(f"Best C: {best_param}")
print(f"Simple CV Score (OPTIMISTIC): {best_score:.4f}")

# ====== Nested CV ======
print("\nNested CV (UNBIASED):")
nested_scores = []

for train_idx, test_idx in KFold(n_splits=5).split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=3,
        scoring='f1'
    )
    
    grid_search.fit(X_train, y_train)
    score = grid_search.score(X_test, y_test)
    nested_scores.append(score)

print(f"Nested CV Score (UNBIASED): {np.mean(nested_scores):.4f} "
      f"(± {np.std(nested_scores):.4f})")

# 差を表示
print(f"\nBias (Simple CV - Nested CV): " 
      f"{best_score - np.mean(nested_scores):.4f}")
```

### 結果例

```
Simple CV (BIASED):
Best C: 0.1
Simple CV Score (OPTIMISTIC): 0.875

Nested CV (UNBIASED):
Nested CV Score (UNBIASED): 0.823 (± 0.035)

Bias (Simple CV - Nested CV): 0.052 ← ← ← 約 5% の楽観性バイアス！
```

---

## 実務的な可視化と検証

### 可視化1：ハイパーパラメータの安定性

```python
import matplotlib.pyplot as plt

# 各 Outer Fold で選ばれたベストハイパーパラメータ
best_params_per_fold = []
fold_scores = []

for train_idx, test_idx in KFold(n_splits=5).split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=3
    )
    
    grid_search.fit(X_train, y_train)
    best_params_per_fold.append(grid_search.best_params_['C'])
    fold_scores.append(grid_search.score(X_test, y_test))

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(5), best_params_per_fold, 'o-')
axes[0].set_xlabel('Outer Fold')
axes[0].set_ylabel('Best C Value')
axes[0].set_yscale('log')
axes[0].set_title('Best Hyperparameter Stability')
ax[0].grid()

axes[1].boxplot(fold_scores)
axes[1].set_ylabel('Test F1 Score')
axes[1].set_title('Nested CV Score Distribution')
axes[1].grid()

plt.tight_layout()
plt.show()

print(f"Hyperparameter consistency: "
      f"Mean Best C = {np.mean(best_params_per_fold):.3f}")
print(f"Score stability: "
      f"Std Dev = {np.std(fold_scores):.3f}")
```

### 可視化2：決定境界の可視化（2次元特徴量の場合）

```python
# 2 つの特徴量でモデルの決定境界を比較

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for fold_idx, (train_idx, test_idx) in enumerate(
    KFold(n_splits=5).split(X[:, :2])  # 最初の2特徴量だけ
):
    X_2d_train, X_2d_test = X[:,:2][train_idx], X[:,:2][test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # ハイパーパラメータチューニング
    grid_search = GridSearchCV(
        LogisticRegression(),
        {'C': [0.1, 1.0, 10.0]},
        cv=3
    )
    grid_search.fit(X_2d_train, y_train)
    
    # 決定境界プロット
    ax = axes[fold_idx // 3, fold_idx % 3]
    
    xx, yy = np.meshgrid(
        np.linspace(X_2d_train[:, 0].min(), X_2d_train[:, 0].max(), 100),
        np.linspace(X_2d_train[:, 1].min(), X_2d_train[:, 1].max(), 100)
    )
    
    Z = grid_search.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=20, alpha=0.6)
    ax.scatter(X_2d_train[y_train == 0, 0], X_2d_train[y_train == 0, 1], 
               c='blue', label='Train 0')
    ax.scatter(X_2d_train[y_train == 1, 0], X_2d_train[y_train == 1, 1],
               c='red', label='Train 1')
    ax.scatter(X_2d_test[y_test == 0, 0], X_2d_test[y_test == 0, 1],
               c='cyan', marker='x', s=100, label='Test 0')
    ax.scatter(X_2d_test[y_test == 1, 0], X_2d_test[y_test == 1, 1],
               c='orange', marker='x', s=100, label='Test 1')
    
    test_score = grid_search.score(X_2d_test, y_test)
    ax.set_title(f'Fold {fold_idx+1} (Score={test_score:.3f})')
    ax.legend()

plt.tight_layout()
plt.show()
```

---

## よくある誤り

### 誤り1：テストセットで複数回チューニング

```python
# ❌ 悪い実装
for hp in hyperparam_list:
    model = train_on_train_set(hp)
    score = model.evaluate_on_test_set()  # テストセット使用して最適化
    
    if score > best:
        best = score
        best_hp = hp

# テストセットが訓練データと同じ役割になる
# → 楽観性バイアスが発生

# ✅ 良い実装
for hp in hyperparam_list:
    model = train_on_train_set(hp)
    score = cross_val_score_on_train_set()  # 訓練セット内のCV
    
    if score > best:
        best = score
        best_hp = hp

# 最終評価は未見テストセットのみ
final_score = model.evaluate_on_test_set()
```

### 誤り2：Nested CV の構造を誤解

```python
# ❌ 悪い構造（チューニングと評価が独立してない）
grid_search.fit(X_train, y_train)
outer_score = cross_val_score(grid_search, X_test, y_test)

# ✅ 良い構造（各 Outer Fold ごとにチューニングを再実施）
for train_idx, test_idx in KFold(5).split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    grid_search = GridSearchCV(...)  # 毎回新規作成
    grid_search.fit(X_train, y_train)  # 訓練セットのみ
    score = grid_search.score(X_test, y_test)  # Outer テストで評価
```

### 誤り3：計算量無視

```python
# ❌ 過度に複雑な設定
nested_cv = cross_validate(
    GridSearchCV(model, param_grid, cv=10),  # Inner CV = 10
    X, y,
    cv=10  # Outer CV = 10
)
# → 10 * 10 * len(param_grid) の学習が必要
# → 実行時間が爆発

# ✅ 実務的なバランス
nested_cv = cross_validate(
    GridSearchCV(model, param_grid, cv=3),  # Inner CV = 3
    X, y,
    cv=5  # Outer CV = 5
)
# → 3 * 5 * len(param_grid) にとどまる
```

---

## 実務での運用判断

### いつ Nested CV を使うべき？

```
✅ 使うべき：
- ハイパーパラメータの自動チューニング（Grid/Random Search）
- 本番環境での真の性能推定が重要
- 複数のモデル候補を比較

❌ 不要な場合：
- ハイパーパラメータがほぼ決まっている（ロジスティック回帰で C=1.0 固定 など）
- 計算量が極端に制限される
- Holdout テストセットが十分に大きい
```

### 計算量とのトレードオフ

```
データサイズ：1000-10000 行
→ Nested CV (5-Outer, 3-Inner) で数秒～数分

データサイズ：100 万+ 行
→ ハイパーパラメータ候補を絞る、または
→ Stratified K-Fold でランダムサンプリング
→ 計算負荷バランス
```


## References

- Nested Cross-Validation: https://scikit-learn.org/stable/modules/cross_validation.html#nested-vs-non-nested-cross-validation
- Cross-Validation Bias: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
- GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

