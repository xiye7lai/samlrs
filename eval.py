from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import math
import json
import pandas as pd
from tqdm import tqdm

from rec_utils import load_config, calculate_fairness_from_topk, calculate_fairness_from_generated, collect_top10_items

config = load_config()

base_model = config['base_model']
lora_weights = config['lora_path']
result_path = config['result_path']
dataset_name = config['dataset_name']

head_generated_path = config['head_generated_path']
tail_generated_path = config['tail_generated_path']

assert base_model, "Please specify a base_model"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    dtype=torch.bfloat16,
    device_map={'': 0}
)
model.eval()

if dataset_name == 'movie':
    with open("/root/samlrs/data/ml-1m-processed/all_movie.json", 'r', encoding='utf-8') as file:
        all_groups = json.load(file)
    all_item_titles = []
    for group in all_groups:
        for movie in group["movies"]:
            all_item_titles.append(movie["movie_name"])
elif dataset_name == 'steam':
    with open("/root/samlrs/data/steam-processed/all_game.json", 'r', encoding='utf-8') as file:
        all_groups = json.load(file)
    all_item_titles = []
    for group in all_groups:
        for item in group["items"]:
            all_item_titles.append(item["item_name"])
elif dataset_name == 'adm':
    with open("/root/samlrs/data/adm-processed/all_music.json", 'r', encoding='utf-8') as file:
        all_groups = json.load(file)
    all_item_titles = []
    for group in all_groups:
        for item in group["items"]:
            all_item_titles.append(item["item_name"])
else:
    raise ValueError(f"Unknown dataset_name: {dataset_name}")

def batch(lst, batch_size=1):
    chunk_size = (len(lst) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield lst[batch_size * i: batch_size * (i + 1)]

def compute_item_embeddings(item_titles, batch_size=16):
    item_emb_chunks = []
    for _, batch_input in tqdm(enumerate(batch(item_titles, batch_size)), total=(len(item_titles)-1)//batch_size + 1, desc="Item Embeddings"):
        enc = tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.cuda()
        attention_mask = enc.attention_mask.cuda()
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        item_emb_chunks.append(hidden_states[-1][:, -1, :].detach().cpu())
    return torch.cat(item_emb_chunks, dim=0)

def load_generated_dataset(path):
    ds = load_dataset("json", data_files=path)
    return ds["train"]

def encode_texts(text_list, batch_size=256):
    pred_emb_chunks = []
    for _, batch_input in tqdm(enumerate(batch(text_list, batch_size)), total=(len(text_list)-1)//batch_size + 1, desc="Predict Embeddings"):
        enc = tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.cuda()
        attention_mask = enc.attention_mask.cuda()
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pred_emb_chunks.append(hidden_states[-1][:, -1, :].detach().cpu())
    return torch.cat(pred_emb_chunks, dim=0)

def evaluate_split(test_data, item_embedding, id2title, title2id, topk_list):
    if len(test_data) == 0:
        return {"NDCG": [0.0]*len(topk_list), "HR": [0.0]*len(topk_list), "size": 0}

    texts = [ex["generated"] for ex in test_data]
    predict_embeddings = encode_texts(texts, batch_size=256).cuda()
    movie_embedding = item_embedding.cuda()

    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)
    rank = dist.argsort(dim=-1)
    ndcg_rank = rank.argsort(dim=-1)

    NDCG, HR = [], []
    for topk in topk_list:
        S = 0.0
        for i in range(len(test_data)):
            target_title = test_data[i]['correct']
            target_id = title2id[target_title]
            rankId = ndcg_rank[i][target_id].item()
            if rankId < topk:
                S += (1 / math.log(rankId + 2))
        ndcg_k = S / len(test_data) / (1 / math.log(2))
        NDCG.append(ndcg_k)

        S = 0
        for i in range(len(test_data)):
            target_title = test_data[i]['correct']
            target_id = title2id[target_title]
            rankId = ndcg_rank[i][target_id].item()
            if rankId < topk:
                S += 1
        HR.append(S / len(test_data))

    return {"NDCG": NDCG, "HR": HR, "size": len(test_data)}

item_embedding = compute_item_embeddings(all_item_titles, batch_size=16)
movie_ids = list(range(len(all_item_titles)))
title2id = dict(zip(all_item_titles, movie_ids))
id2title = dict(zip(movie_ids, all_item_titles))

topk_list = [1, 3, 5, 10, 20]

head_data = load_generated_dataset(head_generated_path)
tail_data = load_generated_dataset(tail_generated_path)

print(f"Loaded head split from: {head_generated_path} (size={len(head_data)})")
print(f"Loaded tail split from: {tail_generated_path} (size={len(tail_data)})")

head_metrics = evaluate_split(head_data, item_embedding, id2title, title2id, topk_list)
print("[HEAD] NDCG:", head_metrics["NDCG"])
print("[HEAD] HR  :", head_metrics["HR"])

tail_metrics = evaluate_split(tail_data, item_embedding, id2title, title2id, topk_list)
print("[TAIL] NDCG:", tail_metrics["NDCG"])
print("[TAIL] HR  :", tail_metrics["HR"])

overall_list = [ex for ex in head_data] + [ex for ex in tail_data]
overall_metrics = evaluate_split(overall_list, item_embedding, id2title, title2id, topk_list)
print("[OVERALL] NDCG:", overall_metrics["NDCG"])
print("[OVERALL] HR  :", overall_metrics["HR"])

result_dict = {
    "topk": topk_list,
    "head": {"size": head_metrics["size"], "NDCG": head_metrics["NDCG"], "HR": head_metrics["HR"]},
    "tail": {"size": tail_metrics["size"], "NDCG": tail_metrics["NDCG"], "HR": tail_metrics["HR"]},
    "overall": {"size": overall_metrics["size"], "NDCG": overall_metrics["NDCG"], "HR": overall_metrics["HR"]},
}

with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)

print("_" * 100)
print("Saved metrics to:", result_path)
