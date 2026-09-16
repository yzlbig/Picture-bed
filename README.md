#!/usr/bin/env python3
"""Evaluate Gemma-4-26B-A4B-IT on AIME2026."""

from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType


_old_request = requests.Session.request


def _request_no_verify(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)


requests.Session.request = _request_no_verify

MODEL = "gemma-4-26b-a4b-it"
API_URL = "http://127.0.0.1:8009/v1"
TP_CONFIG = "tp4"
DATASET_DIR = "/home/yzl/datasets/aime26"

timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
work_dir = (
    f"/home/yzl/precision/{TP_CONFIG}/{MODEL}/aime26/"
    f"{timestamp}_no_thinking_8k"
)

task_cfg = TaskConfig(
    model=MODEL,
    api_url=API_URL,
    api_key="EMPTY",
    eval_type=EvalType.OPENAI_API,
    model_task="text_generation",
    datasets=["aime26"],
    dataset_args={
        "aime26": {
            "dataset_id": DATASET_DIR,
        }
    },
    dataset_hub="modelscope",
    eval_batch_size=16,
    generation_config={
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "n": 1,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    },
    timeout=60000,
    stream=True,
    seed=42,
    work_dir=work_dir,
)

run_task(task_cfg=task_cfg)
