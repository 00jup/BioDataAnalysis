# Codex DILI Experiment

- Feature set: `C`
- Train shape: `(331, 2225)`
- Validation shape: `(72, 2225)`
- Best model by validation AUC: `ExtraTrees_C`
- Best validation AUC: `0.9012`

| Model | CV AUC | Val AUC | Val F1 | Val Precision | Val Recall | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| ExtraTrees_C | 0.9153 | 0.9012 | 0.8378 | 0.8158 | 0.8611 | 0.499 |
| RandomForest_C | 0.9188 | 0.9005 | 0.8235 | 0.8750 | 0.7778 | 0.585 |
| Ensemble_C | 0.9217 | 0.8958 | 0.8235 | 0.8750 | 0.7778 | 0.669 |
| XGBoost_C | 0.9085 | 0.8858 | 0.8354 | 0.7674 | 0.9167 | 0.295 |
| LightGBM_C | 0.9006 | 0.8765 | 0.8395 | 0.7556 | 0.9444 | 0.136 |
| Logistic_C | 0.8852 | 0.8410 | 0.8267 | 0.7949 | 0.8611 | 0.340 |
| SVM_RBF_C | 0.8856 | 0.8364 | 0.7097 | 0.8462 | 0.6111 | 0.786 |
