"""
使用 magpie-ultra 数据集对 pretrain model SFT
"""

import os
import tiktoken
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

ds = load_dataset("argilla/magpie-ultra-v0.1", split="train")

DATA_CACHE_DIR = "magpie_ultra"
if not os.path.exists(DATA_CACHE_DIR):
    os.makedirs(DATA_CACHE_DIR)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]  # end of text token


def tokenizer(example):
    instruct_text = example["instruction"]
    response_text = example["response"]
    # 合并 instruct 和 response
    combined_text = instruct_text + " " + response_text
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(combined_text))
    tokens_np = np.array(tokens, dtype=np.uint16)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "Token indices out of bounds"
    return tokens_np


# 初始化列表，用于存储编码后的结果
encoded_data_train = np.array([], dtype=np.uint16)
encoded_data_test = np.array([], dtype=np.uint16)


nprocs = max(1, mp.cpu_count() - 1)
with mp.Pool(nprocs) as pool:
    progress_bar = None
    count = 0
    for tokens in pool.imap(tokenizer, ds, chunksize=64):
        if count >= ds.num_rows * 0.8:
            encoded_data_test = np.append(encoded_data_test, tokens)
        else:
            encoded_data_train = np.append(encoded_data_train, tokens)
        count += 1
        if progress_bar is None:
            progress_bar = tqdm(total=50000, unit="rows")
        progress_bar.update(1)


np.save(os.path.join(DATA_CACHE_DIR, f"{DATA_CACHE_DIR}_train"), encoded_data_train)
np.save(os.path.join(DATA_CACHE_DIR, f"{DATA_CACHE_DIR}_val"), encoded_data_test)
