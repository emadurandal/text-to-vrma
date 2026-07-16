# build_text_encoder_base.py — gated meta-llama の代わりに公開ミラーから
# LLM2Vec ベースモデル (Llama-3-8B-Instruct + mntp アダプタをマージ済み) を構築する。
#
# 通常フロー (要gated承認):
#   LlamaBiModel.from_pretrained("McGill-NLP/...-mntp")
#     → transformers が adapter_config の base (meta-llama/Meta-Llama-3-8B-Instruct) をDL
#     → mntp アダプタ適用・マージ → その上に supervised アダプタ
# ここでは base を NousResearch ミラー (Llama 3 ライセンスに基づく公開再配布) に差し替えて
# 同じマージ結果をローカルに保存する。supervised アダプタは実行時に通常通り適用される。
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from ardy.model.llm2vec.models.bidirectional_llama import LlamaBiModel

MIRROR = "NousResearch/Meta-Llama-3-8B-Instruct"
MNTP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="マージ済みモデルの保存先ディレクトリ")
ap.add_argument("--base", default=MIRROR, help="ベースモデル (既定: NousResearchミラー)")
args = ap.parse_args()
OUT = args.out
MIRROR = args.base

print(f"loading base from mirror: {MIRROR}")
model = LlamaBiModel.from_pretrained(MIRROR, dtype=torch.bfloat16)
print("applying mntp adapter...")
model = PeftModel.from_pretrained(model, MNTP)
model = model.merge_and_unload()

print(f"saving merged model to {OUT}")
model.save_pretrained(OUT, safe_serialization=True)
# トークナイザは mntp リポジトリのもの (special tokens 設定込み) を使う
AutoTokenizer.from_pretrained(MNTP).save_pretrained(OUT)

# vendored llm2vec は config の _name_or_path で Llama-3-8B-Instruct 用の
# 命令フォーマットを選ぶため、正式な名前を書き込んでおく
cfg_path = os.path.join(OUT, "config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["_name_or_path"] = "meta-llama/Meta-Llama-3-8B-Instruct"
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print("done")
