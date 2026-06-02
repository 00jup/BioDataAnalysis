"""ChemBERTa 파인튜닝 — DILI(vivo) 간독성 분류.

사전학습 분자 트랜스포머(DeepChem/ChemBERTa-77M-MLM, ~77M 분자 MLM 학습)를
DILI 데이터로 파인튜닝한다. chemprop v31 과 '같은 scaffold split' 을 써서 공정 비교한다.

Colab 사용법:
    1) 런타임 → 하드웨어 가속기 → GPU (T4)
    2) !pip -q install -U transformers accelerate datasets scikit-learn rdkit
    3) all.csv, splits.json 두 파일 업로드 (data/chemprop_scaffold_v3/vivo/)
    4) !python chemberta_finetune.py
시간: T4 약 30~50분 (10 epoch), A100 약 5~8분.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# 표준 구조(RoBERTa) 모델만 사용 — trust_remote_code/버전핀/재시작 전부 불필요.
# 한 줄만 바꾸면 다른 모델로 교체된다:
#   "seyonec/ChemBERTa-zinc-base-v1"  : RoBERTa-base(~큼), ZINC 사전학습 (기본)
#   "Derify/ChemBERTa-druglike"       : drug-like 분자 전용 (DILI에 적합)
#   "DeepChem/ChemBERTa-77M-MTR"      : property 사전학습(작지만 강함)
MODEL_NAME = "Derify/ChemBERTa-druglike"
ALL_CSV = "all.csv"          # data/chemprop_scaffold_v3/vivo/all.csv
SPLITS_JSON = "splits.json"  # 같은 폴더
MAX_LEN = 200
EPOCHS = 10
LR = 2e-5
BATCH = 32                   # base 크기 모델 → 32 (OOM 시 16)

# ---------- 1. 데이터 + 같은 scaffold split ----------
df = pd.read_csv(ALL_CSV)
sp = json.load(open(SPLITS_JSON))
sp = sp[0] if isinstance(sp, list) else sp
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
# 모델 최대 길이에 맞춰 MAX_LEN 자동 축소 (position 인덱스 초과 → CUDA assert 방지)
_cfg = AutoConfig.from_pretrained(MODEL_NAME)
MAX_LEN = min(MAX_LEN, getattr(_cfg, "max_position_embeddings", MAX_LEN + 2) - 2)
print(f"MAX_LEN={MAX_LEN}, tokenizer vocab={len(tok)}")


def make(idx):
    sub = df.iloc[idx].reset_index(drop=True)
    enc = tok(sub["canonical_smiles"].tolist(), truncation=True, max_length=MAX_LEN)
    enc["labels"] = sub["label"].astype(int).tolist()
    return _DS(enc)


class _DS(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["labels"])

    def __getitem__(self, i):
        # raw 리스트/정수 반환 → DataCollatorWithPadding 가 패딩·텐서화 담당
        return {k: v[i] for k, v in self.enc.items()}


ds_tr, ds_va, ds_te = make(sp["train"]), make(sp["val"]), make(sp["test"])
y_tr = np.array(df.iloc[sp["train"]]["label"])
pos_w = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)  # 양성 가중치(불균형)
print(f"train {len(ds_tr)} / val {len(ds_va)} / test {len(ds_te)}  pos_weight={pos_w:.2f}")

# ---------- 2. 모델 (사전학습 가중치 + 분류 헤드) ----------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.resize_token_embeddings(len(tok))  # 임베딩을 토크나이저 vocab 에 맞춤 (token id 초과 → CUDA assert 방지)


# ---------- 3. 클래스 가중 손실 ----------
class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        weight = torch.tensor([1.0, pos_w], device=out.logits.device, dtype=out.logits.dtype)
        loss = nn.CrossEntropyLoss(weight=weight)(out.logits, labels)
        return (loss, out) if return_outputs else loss


def metrics(p):
    logits, labels = p
    prob = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    pred = (prob >= 0.5).astype(int)
    return {"mcc": matthews_corrcoef(labels, pred), "auc": roc_auc_score(labels, prob)}


args = TrainingArguments(
    output_dir="chemberta_dili",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=128,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mcc",
    greater_is_better=True,
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
)
trainer = WTrainer(
    model=model,
    args=args,
    train_dataset=ds_tr,
    eval_dataset=ds_va,
    compute_metrics=metrics,
    data_collator=DataCollatorWithPadding(tok),
)
trainer.train()

# ---------- 4. 예측 추출(val+test) + Drive 영구 저장 ----------
import os

SAVE_DIR = "/content/drive/MyDrive/dili_models" if os.path.isdir("/content/drive/MyDrive") else "./dili_models"
os.makedirs(SAVE_DIR, exist_ok=True)
TAG = MODEL_NAME.split("/")[-1]
print(f"\n저장 위치: {SAVE_DIR}  ({'Drive 영구' if 'drive' in SAVE_DIR else '로컬 휘발성 — Drive 마운트 권장'})")


def export(ds, split):
    """split 의 (canonical_smiles, label, prob) CSV 저장 — stacking 용."""
    pr = trainer.predict(ds)
    prob = torch.softmax(torch.tensor(pr.predictions), dim=1)[:, 1].numpy()
    sub = df.iloc[sp[split]].reset_index(drop=True)
    out = pd.DataFrame(
        {"canonical_smiles": sub["canonical_smiles"], "label": sub["label"].astype(int), "prob": prob}
    )
    path = os.path.join(SAVE_DIR, f"{TAG}_{split}_pred.csv")
    out.to_csv(path, index=False)
    print(f"  저장: {path}")
    return out


val_out = export(ds_va, "val")
test_out = export(ds_te, "test")
prob = test_out["prob"].to_numpy()
y = test_out["label"].to_numpy()
trainer.save_model(os.path.join(SAVE_DIR, f"{TAG}_best"))

# ---------- 5. test threshold sweep + 비교 ----------
print(f"\n=== TEST (scaffold-OOD) ===  AUC {roc_auc_score(y, prob):.3f}")
print(f"{'thr':>5}{'TPR':>7}{'TNR':>7}{'MCC':>8}")
best = (-1, 0.5)
for t in np.linspace(0.1, 0.9, 17):
    pred = (prob >= t).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]
    fp, tn = cm[1]
    mcc = matthews_corrcoef(y, pred)
    if mcc > best[0]:
        best = (mcc, t)
    print(f"{t:>5.2f}{tp / max(tp + fn, 1):>7.3f}{tn / max(fp + tn, 1):>7.3f}{mcc:>+8.3f}")
print(f"\nchemprop v31 (= v27 레시피): AUC 0.788 / MCC 0.436 | {TAG} best MCC {best[0]:+.3f} @ {best[1]:.2f}")
