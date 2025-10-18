import json
import os
import time
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
)

import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from rec_utils import load_config

config = load_config()

base_model = config['base_model']
lora_weights = config['lora_path']
dataset_name = config['dataset_name']
train_path = config['train_path']
val_path = config['val_path']
use_weight = config['use_weight']
epoch = config['epoch']

rho = float(config.get('rho', 0.05))
lambda_coef = float(config.get('lambda', 0.5))


class EISAMTrainer(transformers.Trainer):


    def __init__(
        self,
        rho: float = rho,
        lambda_coef: float = lambda_coef,
        use_weight: bool = use_weight,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rho = float(rho)
        self.lambda_coef = float(lambda_coef)
        self.use_weight = bool(use_weight)

    def _xe_per_token(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        losses = loss_fct(shift_logits, shift_labels)
        return losses

    def compute_Ly(self, model, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """(scalar_loss, per_sample_loss)"""
        weights = inputs.get("weight", None)
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "weight"})
        logits = outputs.get("logits")
        losses = self._xe_per_token(logits, labels)  # [B*T]
        B = labels.size(0)
        losses = losses.view(B, -1).mean(dim=1)      # [B]
        if self.use_weight and (weights is not None):
            weights = weights.to(losses.device)
            loss = (losses * weights).mean()
        else:
            loss = losses.mean()
        return loss, losses.detach()

    def compute_Lg(self, model, inputs) -> torch.Tensor:
        loss_y, _ = self.compute_Ly(model, inputs)
        return loss_y

    # ---------- EISAM ----------
    @torch.no_grad()
    def _grad_norm(self, params) -> torch.Tensor:
        norms = []
        for p in params:
            if p.grad is not None:
                norms.append(p.grad.norm(p=2))
        if len(norms) == 0:
            return torch.tensor(0.0, device=params[0].device)
        return torch.norm(torch.stack(norms), p=2)

    def _zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ----- Step 1/2：compute ε(θ) -----
        self._zero_grad()
        Lg = self.compute_Lg(model, inputs)
        Lg.backward()
        grad_norm = self._grad_norm(model.parameters())
        scale = (self.rho / (grad_norm + 1e-12)).to(next(model.parameters()).device)

        eps_list = []
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    eps_list.append(None)
                    continue
                e = p.grad * scale
                p.add_(e)
                eps_list.append(e)

        # ----- Step 3：compute g1 = ∂ L_y / ∂θ |_{θ+ε} -----
        self._zero_grad()
        Ly_perturbed, _ = self.compute_Ly(model, inputs)
        Ly_perturbed.backward()
        g1 = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]

        with torch.no_grad():
            for p, e in zip(model.parameters(), eps_list):
                if e is not None:
                    p.sub_(e)

        # ----- Step 4：compute g2 = ∂ [L_y - λ L_g] / ∂θ -----
        self._zero_grad()
        Ly_orig, _ = self.compute_Ly(model, inputs)
        Lg_orig = self.compute_Lg(model, inputs)
        objective = Ly_orig - self.lambda_coef * Lg_orig
        objective.backward()
        g2 = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]

        # ----- Step 5：compute λ g1 + g2, write  p.grad -----
        self._zero_grad()
        with torch.no_grad():
            for p, gg1, gg2 in zip(model.parameters(), g1, g2):
                if gg1 is None and gg2 is None:
                    continue
                if gg1 is None:
                    p.grad = gg2
                elif gg2 is None:
                    p.grad = self.lambda_coef * gg1
                else:
                    p.grad = self.lambda_coef * gg1 + gg2

        loss_for_log = Ly_orig.detach()
        return loss_for_log / self.args.gradient_accumulation_steps


def train(
        base_model: str = base_model,
        train_data_path: str = train_path,
        val_data_path: str = val_path,
        output_dir: str = lora_weights,
        sample: int = -1,
        seed: int = 0,
        tau: int = 1,
        # training hyperparams
        batch_size: int = 64,
        micro_batch_size: int = 16,
        num_epochs: int = epoch,
        learning_rate: float = 1e-3,
        cutoff_len: int = 512,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        eval_sample: int = -1,
        use_weight: bool = use_weight,
        rho: float = rho,
        lambda_coef: float = lambda_coef,
):
    print(
        f"Training (EISAM) with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"eval_sample: {eval_sample}\n"
        f"use_weight: {use_weight}\n"
        f"rho: {rho}\n"
        f"lambda: {lambda_coef}\n"
    )
    assert base_model, "Please specify a base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    EOS_TOKEN = tokenizer.eos_token

    config_lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config_lora)

    def generate_prompt(data_point, eos_token):
        if data_point.get("input"):
            data_point["text"] = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}""" + eos_token
            return data_point
        else:
            data_point["text"] = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}""" + eos_token
            return data_point

    if train_data_path.endswith(".json"):
        train_data = load_dataset("json", data_files=train_data_path)
        train_data = train_data.map(lambda x: generate_prompt(x, EOS_TOKEN))

        def tokenize_dataset(dataset):
            return tokenizer(dataset["text"], truncation=True, max_length=cutoff_len, padding=False)

        def make_labels(x):
            x["labels"] = x["input_ids"].copy()
            return x

        train_data = train_data.map(tokenize_dataset, batched=True)
        train_data = train_data.map(make_labels, batched=True)
        train_data = train_data["train"].shuffle(seed=seed)

    if val_data_path.endswith(".json"):
        val_data = load_dataset("json", data_files=val_data_path)
        val_data = val_data.map(lambda x: generate_prompt(x, EOS_TOKEN))
        val_data = val_data["train"].shuffle(seed=seed)
    else:
        val_data = None

    print(len(train_data))
    print(train_data[0])

    model.print_trainable_parameters()

    trainer = EISAMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=8,
            optim="adamw_8bit",
            save_strategy="epoch",
            seed=seed,
            output_dir=output_dir,
            remove_unused_columns=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        rho=rho,
        lambda_coef=lambda_coef,
        use_weight=use_weight,
    )

    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    train()
