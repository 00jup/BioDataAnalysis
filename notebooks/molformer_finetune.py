"""MolFormer-XL 파인튜닝 — DILI(vivo) 간독성 분류 (큰 모델).

사전학습 분자 트랜스포머를 DILI 데이터로 파인튜닝한다.
ChemBERTa(작음) 대비 MolFormer-XL(~44M)은 훨씬 크고 강하다.
chemprop v31(= v27 동일 하이퍼파라미터, 같은 scaffold_v3 split)과 공정 비교.

MODEL_NAME 만 바꾸면 다른 인코더도 동일 코드로 학습된다:
  - "ibm-research/MoLFormer-XL-both-10pct"  (큰 모델, trust_remote_code 필요)
  - "DeepChem/ChemBERTa-77M-MTR"            (property 사전학습, 작음)
  - "DeepChem/ChemBERTa-77M-MLM"            (작음)

분류 헤드가 없는 모델도 되도록 AutoModel(인코더)+직접 헤드(mean-pool→Linear) 방식.

Colab: ① GPU 켜기  ② pip 설치  ③ all.csv·splits.json 업로드  ④ 본 스크립트 실행
시간: MolFormer-XL T4 약 20~40분.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# =====================================================================
# 호환 shim: transformers 를 낮추지 않고(=충돌 0), MolFormer 원격코드가
# 찾는 옛 심볼만 현재 transformers 에 직접 주입한다. (다운그레이드/재시작 불필요)
# =====================================================================
import sys
import types

# 1) transformers.onnx.OnnxConfig (config 가 import) — 학습에 안 쓰이므로 빈 stub
try:
    from transformers.onnx import OnnxConfig  # noqa: F401
except Exception:
    _m = types.ModuleType("transformers.onnx")

    class OnnxConfig:  # stub
        pass

    _m.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = _m

# 2) transformers.pytorch_utils 의 옛 함수들 (modeling 이 import) — 실제 구현 주입
import transformers.pytorch_utils as _pu  # noqa: E402

if not hasattr(_pu, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

if not hasattr(_pu, "prune_linear_layer"):
    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            b = layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        nl = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        nl.weight.requires_grad = False
        nl.weight.copy_(W.contiguous())
        nl.weight.requires_grad = True
        if layer.bias is not None:
            nl.bias.requires_grad = False
            nl.bias.copy_(b.contiguous())
            nl.bias.requires_grad = True
        return nl

    _pu.prune_linear_layer = prune_linear_layer

if not hasattr(_pu, "apply_chunking_to_forward"):
    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        return forward_fn(*input_tensors)  # chunking 없이 그대로 (결과 동일)

    _pu.apply_chunking_to_forward = apply_chunking_to_forward

MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
TRUST = True          # 커스텀 코드 모델(MolFormer 등) 허용
ALL_CSV = "all.csv"
SPLITS_JSON = "splits.json"
MAX_LEN = 202
EPOCHS = 10
LR = 3e-5
BATCH = 16            # fp32(아래) 메모리 안전 (OOM 시 8)

# ---------- 1. 데이터 + 같은 scaffold split ----------
df = pd.read_csv(ALL_CSV)
sp = json.load(open(SPLITS_JSON))
sp = sp[0] if isinstance(sp, list) else sp
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=TRUST)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.sep_token or "[PAD]"


class DS(torch.utils.data.Dataset):
    def __init__(self, idx):
        sub = df.iloc[idx].reset_index(drop=True)
        self.enc = tok(sub["canonical_smiles"].tolist(), truncation=True, max_length=MAX_LEN)
        self.enc["labels"] = sub["label"].astype(int).tolist()

    def __len__(self):
        return len(self.enc["labels"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


ds_tr, ds_va, ds_te = DS(sp["train"]), DS(sp["val"]), DS(sp["test"])
y_tr = np.array(df.iloc[sp["train"]]["label"])
pos_w = float((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
print(f"train {len(ds_tr)} / val {len(ds_va)} / test {len(ds_te)}  pos_weight={pos_w:.2f}  model={MODEL_NAME}")


# ---------- 2. 인코더 + 직접 분류 헤드 (mean-pool) ----------
class EncoderClassifier(nn.Module):
    def __init__(self, name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name, trust_remote_code=TRUST)
        # 최신 transformers 에서 빠진 get_head_mask 를 인코더에 주입 (학습 시 head_mask=None)
        if not hasattr(self.encoder, "get_head_mask"):
            import types as _t

            def _ghm(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                return head_mask.to(dtype=self.dtype)

            self.encoder.get_head_mask = _t.MethodType(_ghm, self.encoder)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hid, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                          # (B, L, H)
        m = attention_mask.unsqueeze(-1).type_as(h)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)  # masked mean pool
        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(logits=logits)


model = EncoderClassifier(MODEL_NAME)


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
    prob = torch.softmax(torch.tensor(logits).float(), dim=1)[:, 1].numpy()
    prob = np.nan_to_num(prob, nan=0.5)  # 혹시 모를 NaN 방어
    return {
        "mcc": matthews_corrcoef(labels, (prob >= 0.5).astype(int)),
        "auc": roc_auc_score(labels, prob),
    }


args = TrainingArguments(
    output_dir="mol_dili",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=64,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=False,                  # MolFormer 는 fp16 에서 NaN → fp32 로 안정화
    eval_strategy="epoch",       # 현재 Colab transformers(최신) 기준
    save_strategy="no",          # 커스텀 모듈 저장 이슈 회피 (마지막에 수동 저장)
    logging_steps=50,
    report_to="none",
)
trainer = WTrainer(
    model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
    compute_metrics=metrics, data_collator=DataCollatorWithPadding(tok),
)
trainer.train()

# ---------- 4. 예측 추출(val+test) + Drive 영구 저장 ----------
import os

# Drive 가 마운트돼 있으면 거기에(영구), 아니면 로컬(/content, 휘발성)에 저장
SAVE_DIR = "/content/drive/MyDrive/dili_models" if os.path.isdir("/content/drive/MyDrive") else "./dili_models"
os.makedirs(SAVE_DIR, exist_ok=True)
TAG = MODEL_NAME.split("/")[-1]
print(f"\n저장 위치: {SAVE_DIR}  ({'Drive 영구' if 'drive' in SAVE_DIR else '로컬 휘발성 — Drive 마운트 권장'})")


def export(ds, split):
    """split 의 (canonical_smiles, label, prob) CSV 저장 — stacking 용."""
    pr = trainer.predict(ds)
    prob = torch.softmax(torch.tensor(pr.predictions).float(), dim=1)[:, 1].numpy()
    prob = np.nan_to_num(prob, nan=0.5)  # NaN 방어
    sub = df.iloc[sp[split]].reset_index(drop=True)  # ds 와 동일 순서
    out = pd.DataFrame(
        {"canonical_smiles": sub["canonical_smiles"], "label": sub["label"].astype(int), "prob": prob}
    )
    path = os.path.join(SAVE_DIR, f"{TAG}_{split}_pred.csv")
    out.to_csv(path, index=False)
    print(f"  저장: {path}")
    return out


val_out = export(ds_va, "val")   # stacking α/threshold 결정용
test_out = export(ds_te, "test")  # 최종 평가용
prob = test_out["prob"].to_numpy()
y = test_out["label"].to_numpy()

try:
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{TAG}_weights.pt"))
    print(f"  저장: {SAVE_DIR}/{TAG}_weights.pt (가중치)")
except Exception as e:
    print("  가중치 저장 건너뜀:", e)

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

print("\n=== 비교 (같은 scaffold_v3 split) ===")
print("chemprop v31 (= v27 레시피)  : AUC 0.788  MCC 0.436")
print(f"{TAG:<28}: AUC {roc_auc_score(y, prob):.3f}  best MCC {best[0]:+.3f} @ {best[1]:.2f}")
