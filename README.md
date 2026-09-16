#!/usr/bin/env python3
"""Evaluate Gemma-4-26B-A4B-IT on MMMU-Pro.

Default: Math subset (60 samples).
Set MMMU_SUBSET=all for all 29 subsets (1730 samples).
"""

import glob
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType


MODEL = "gemma-4-26b-a4b-it"
API_URL = "http://127.0.0.1:8009/v1"
TP_CONFIG = "tp4"
DATASET_DIR = "/home/yzl/datasets/mmmu_pro"

SUBSET = os.environ.get("MMMU_SUBSET", "Math")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TOP_K = int(os.environ.get("TOP_K", "64"))
THINKING = os.environ.get("THINKING", "0") == "1"

if SUBSET.lower() == "all":
    import pandas as pd

    parquet_files = sorted(glob.glob(f"{DATASET_DIR}/*.parquet"))
    dataframe = pd.concat([pd.read_parquet(path) for path in parquet_files])
    subset_column = "subset" if "subset" in dataframe.columns else "subject"
    subset_list = sorted(dataframe[subset_column].unique().tolist())
    subset_tag = "all"
else:
    subset_list = [SUBSET]
    subset_tag = SUBSET.lower().replace(" ", "_")

timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
thinking_tag = "thinking" if THINKING else "no_thinking"
work_dir = (
    f"/home/yzl/precision/{TP_CONFIG}/{MODEL}/mmmu_pro/"
    f"{timestamp}_{subset_tag}_{thinking_tag}_8k"
)

task_cfg = TaskConfig(
    model=MODEL,
    api_url=API_URL,
    api_key="EMPTY",
    eval_type=EvalType.OPENAI_API,
    model_task="text_generation",
    datasets=["mmmu_pro"],
    dataset_args={
        "mmmu_pro": {
            "dataset_id": DATASET_DIR,
            "extra_params": {"dataset_format": "default"},
            "subset_list": subset_list,
        }
    },
    dataset_hub="modelscope",
    eval_batch_size=CONCURRENCY,
    generation_config={
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "n": 1,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": THINKING,
            }
        },
    },
    timeout=60000,
    stream=True,
    seed=42,
    work_dir=work_dir,
)

print(
    f"[MMMU-Pro] model={MODEL} subsets={len(subset_list)} "
    f"subset={SUBSET} concurrency={CONCURRENCY} max_tokens={MAX_TOKENS} "
    f"temperature={TEMPERATURE} thinking={THINKING}",
    flush=True,
)
run_task(task_cfg=task_cfg)
