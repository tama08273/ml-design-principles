# ML Engineering Insights

## About

このリポジトリは、機械学習エンジニアが実務を通じて学んだ、参考書や教科書には載らない知見をまとめたものです。

- 金融データサイエンス（預金残高計、不正検知）
- 自然言語処理の実運用（NLP分類、KIBIT開発）
- モデル選択と評価指標の実践的理解

## Contents

### 001: 外れ値除去の誤解
[001_outlier_removal_misconception.md](insights/001_outlier_removal_misconception.md)

金融データのべき分布では、外れ値が重要な知見を持っている。単純な除去ではなく、対数変換による分布特性の保持が重要という学習。

### 002: 不均衡データと閾値調整
[002_imbalanced_data_threshold_tuning.md](insights/002_imbalanced_data_threshold_tuning.md)

実務での分類問題はその多くが不均衡データ。精度偏重ではなく、ビジネス要件に応じた評価指標選択（Precision/Recall/F1）と閾値調整が不可欠。

### 003: モデル複雑性と選択基準
[003_model_complexity_tradeoff.md](insights/003_model_complexity_tradeoff.md)

複雑 ≠ 正解。ロジスティック回帰から決定木、勾配ブースティングまで、データ複雑性と要件に応じた使い分けが必要。解釈性と精度のトレードオフを理解することが重要。

### 004: クロスバリデーションの楽観性バイアス
[004_cross_validation_optimism_bias.md](insights/004_cross_validation_optimism_bias.md)

クロスバリデーションは訓練時評価では楽観的になりやすい。ホールドアウト+Nested CVおよび可視化による検証で、真の汎化性能を推定することが重要。

### 005: NLPでの過学習と多層防御
[005_nlp_overfitting_multilayer_defense.md](insights/005_nlp_overfitting_multilayer_defense.md)

自然言語処理（KIBIT）での過学習リスク。ルールベース判定（辞書マッピング）とML（Neural Network/Transformer）を組み合わせた多層防御で、誤判定を大幅削減。

---

## 背景

これらの知見は以下の実務経験に基づいています：

- **金融機関でのデータサイエンス**：顧客データ、不正検知、リスク評価
- **自然言語処理の本番運用**：KIBIT（コンプライアンス系NLP MLツール）の開発・運用
- **複数データセットの評価実装**：単純なデータから複雑性の高いデータまで

## 目的

機械学習エンジニアが現場で直面する問題は、教科書とは異なります。

- 「正解データが圧倒的に少ない」
- 「外れ値は本来含まれるべき情報」  
- 「モデル精度が高くても本番では振るわない」
- 「複雑なモデルが必ず最適ではない」

このリポジトリは、そうした「実務の現実」を記録し、
同様の課題に直面するエンジニアへの知見共有を目的としています。

---

## Notes

- これらのドキュメントは**特定分野での実務経験**に基づいています
- 全ての状況に普遍的ではなく、データ特性・ビジネス要件により変動します
- **Feedback / Discussion を歓迎**します

---

## Author

Data Scientist → ML Engineer → AI Security Engineer 志向
