# train_twostage_qwen2vl.py
# 两阶段训练：先纯文本(JSON: instruction/input/output/system)，再图文(RSVQA conversations+vision)
# 并修正评估：不再被 yes/no 误导，长文本用包含/Token-F1/数字特判综合计分
#
# 运行示例（单机多卡）：
#   CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train_twostage_qwen2vl.py \
#     --base_model_dir ./Qwen/Qwen2-VL-2B-Instruct \
#     --text_json_path data_text.json \
#     --vl_json_path data_vl.json \
#     --output_dir ./output/Qwen2-VL-2B
#
# 运行示例（单卡）：
#   CUDA_VISIBLE_DEVICES=4 python train_twostage_qwen2vl.py --text_json_path data_text.json --vl_json_path data_vl.json

import os
import re
import json
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    TrainingArguments,
    Trainer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
from swanlab.integration.transformers import SwanLabCallback
import swanlab


# --------------------------
# 评估：文本归一 + yes/no 特判 + 数字特判 + token-F1
# --------------------------
def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[，,。\.！？!\?；;：:\(\)\[\]{}\"'“”‘’、/\\|<>《》]", "", s)
    return s

def _yesno_map(s: str) -> Optional[str]:
    s0 = _norm_text(s)
    if not s0:
        return None
    yes_set = {"yes", "y", "是", "对", "正确", "可以", "有", "属于", "存在", "能"}
    no_set  = {"no", "n", "否", "不", "错误", "不是", "没有", "不属于", "不存在", "不能"}
    head = s0.split(" ")[0]
    if head in yes_set:
        return "yes"
    if head in no_set:
        return "no"
    if any(t in s0 for t in yes_set):
        return "yes"
    if any(t in s0 for t in no_set):
        return "no"
    return None

def _token_f1(pred: str, gold: str) -> float:
    p = _norm_text(pred)
    g = _norm_text(gold)
    if not p or not g:
        return 0.0
    p_tokens = p.split(" ")
    g_tokens = g.split(" ")
    p_cnt, g_cnt = {}, {}
    for t in p_tokens:
        p_cnt[t] = p_cnt.get(t, 0) + 1
    for t in g_tokens:
        g_cnt[t] = g_cnt.get(t, 0) + 1
    common = 0
    for t, c in p_cnt.items():
        if t in g_cnt:
            common += min(c, g_cnt[t])
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

def is_correct(pred: str, gold: str, f1_threshold: float = 0.6) -> bool:
    # 1) yes/no 特判
    gy = _yesno_map(gold)
    if gy is not None:
        py = _yesno_map(pred)
        return py == gy

    # 2) 数字特判：gold 有数字就优先比数字
    gnum = re.search(r"-?\d+", gold or "")
    if gnum:
        pnum = re.search(r"-?\d+", pred or "")
        return (pnum is not None) and (pnum.group(0) == gnum.group(0))

    # 3) 长文本：包含关系优先，其次 token-F1
    p = _norm_text(pred)
    g = _norm_text(gold)
    if g and (g in p):
        return True
    return _token_f1(pred, gold) >= f1_threshold


# --------------------------
# 混合 Collator（同一 batch 内要么全有视觉字段，要么全没有）
# 两阶段训练天然满足这个要求，所以非常稳。
# --------------------------
@dataclass
class MixedVLTextCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ---- helper: 确保是 1D LongTensor ----
        def to_1d_long(x):
            if isinstance(x, torch.Tensor):
                return x.long()
            return torch.tensor(x, dtype=torch.long)

        input_ids_list = [to_1d_long(f["input_ids"]) for f in features]
        attn_list = [to_1d_long(f["attention_mask"]) for f in features]
        labels_list = [to_1d_long(f["labels"]) for f in features]

        # pad input_ids/attention_mask
        batch = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attn_list},
            padding=True,
            return_tensors="pt",
        )

        # pad labels 到 -100
        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            L = lab.numel()
            padded_labels[i, :L] = lab[:max_len]
        batch["labels"] = padded_labels

        # 视觉字段：可能也是 list/tensor，统一转 tensor
        has_vision = all(("pixel_values" in f and "image_grid_thw" in f) for f in features)
        if has_vision:
            def to_tensor(x, dtype=None):
                if isinstance(x, torch.Tensor):
                    return x if dtype is None else x.to(dtype)
                t = torch.tensor(x)
                return t if dtype is None else t.to(dtype)

            batch["pixel_values"] = torch.stack(
                [to_tensor(f["pixel_values"], dtype=torch.float32) for f in features],
                dim=0
            )
            batch["image_grid_thw"] = torch.stack(
                [to_tensor(f["image_grid_thw"], dtype=torch.long) for f in features],
                dim=0
            )

        return batch

# --------------------------
# 模型/处理器
# --------------------------
def build_tokenizer_processor(base_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_dir)
    return tokenizer, processor

def build_model(base_model_dir: str):
    # 稳定优先：禁用 flash-attn/xformers 路径
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",  # 如果你本地 transformers 不支持，可删掉该参数
    )
    # 配合 gradient checkpointing，避免 requires_grad warning/断梯度
    model.enable_input_require_grads()
    return model


# --------------------------
# 数据预处理：纯文本
# --------------------------
def make_process_text_func(tokenizer, processor, max_length: int):
    def process_text_func(example):
        instruction = (example.get("instruction") or "").strip()
        inp = (example.get("input") or "").strip()
        output = (example.get("output") or "").strip()
        system = (example.get("system") or "").strip()

        user_text = instruction if not inp else f"{instruction}\n{inp}"

        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        response = tokenizer(output, add_special_tokens=False, return_tensors="pt")

        input_ids = torch.cat(
            [inputs["input_ids"][0], response["input_ids"][0], torch.tensor([tokenizer.pad_token_id])],
            dim=0,
        )
        attention_mask = torch.cat(
            [inputs["attention_mask"][0], response["attention_mask"][0], torch.tensor([1])],
            dim=0,
        )
        labels = torch.cat(
            [
                torch.full((inputs["input_ids"].shape[1],), -100, dtype=torch.long),
                response["input_ids"][0],
                torch.tensor([tokenizer.pad_token_id]),
            ],
            dim=0,
        )

        if input_ids.numel() > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return process_text_func


# --------------------------
# 数据预处理：图文 RSVQA conversations 格式
# --------------------------
def make_process_vl_func(tokenizer, processor, max_length: int, resized_hw: int):
    def process_vl_func(example):
        conversation = example["conversations"]
        input_content = conversation[0]["value"]
        output_content = conversation[1]["value"]

        # 格式: "问题文本 <|vision_start|>图像路径<|vision_end|>"
        if "<|vision_start|>" in input_content and "<|vision_end|>" in input_content:
            parts = input_content.split("<|vision_start|>")
            question_text = parts[0].strip()
            file_path = parts[1].split("<|vision_end|>")[0].strip()
        else:
            file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0].strip()
            question_text = "请描述这张遥感图像。"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path, "resized_height": resized_hw, "resized_width": resized_hw},
                    {"type": "text", "text": question_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        response = tokenizer(output_content, add_special_tokens=False, return_tensors="pt")

        input_ids = torch.cat(
            [inputs["input_ids"][0], response["input_ids"][0], torch.tensor([tokenizer.pad_token_id])],
            dim=0,
        )
        attention_mask = torch.cat(
            [inputs["attention_mask"][0], response["attention_mask"][0], torch.tensor([1])],
            dim=0,
        )
        labels = torch.cat(
            [
                torch.full((inputs["input_ids"].shape[1],), -100, dtype=torch.long),
                response["input_ids"][0],
                torch.tensor([tokenizer.pad_token_id]),
            ],
            dim=0,
        )

        if input_ids.numel() > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        image_grid_thw = inputs["image_grid_thw"].squeeze(0)
        pixel_values = inputs["pixel_values"]  # 通常 [1, ...]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    return process_vl_func


# --------------------------
# 推理与评估（只在主进程做）
# --------------------------
@torch.inference_mode()
def predict(messages, model, processor, max_new_tokens: int = 128):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]


def evaluate_vl_dataset(model_to_use, label: str, dataset_list, processor,
                        max_samples: int = 200, f1_threshold: float = 0.6,
                        force_explain: bool = False):
    total, correct = 0, 0
    preds_for_log = []

    for item in dataset_list[:max_samples]:
        input_image_prompt = item["conversations"][0]["value"]
        expected_answer = item["conversations"][1]["value"]

        if "<|vision_start|>" in input_image_prompt and "<|vision_end|>" in input_image_prompt:
            parts = input_image_prompt.split("<|vision_start|>")
            question_text = parts[0].strip()
            origin_image_path = parts[1].split("<|vision_end|>")[0].strip()
        else:
            origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0].strip()
            question_text = "请描述这张遥感图像。"

        if force_explain:
            question_text = question_text + "。请用完整句子回答，并给出简要解释。"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": question_text},
            ]}]

        pred = predict(messages, model_to_use, processor, max_new_tokens=128)
        total += 1
        if is_correct(pred, expected_answer, f1_threshold=f1_threshold):
            correct += 1

        if total <= 5:
            print(f"\n[{label}] Q: {question_text}")
            print(f"GT: {expected_answer}")
            print(f"P : {pred[:400]}")

        preds_for_log.append(swanlab.Image(origin_image_path, caption=f"{label}: {pred[:200]}"))

    acc = correct / total if total else 0.0
    print(f"\n[{label}] Accuracy (hybrid metric): {correct}/{total} = {acc:.2%}")
    return acc, preds_for_log


# --------------------------
# 主流程：两阶段训练
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", default="./Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--text_json_path", required=True, help="纯文本 json，格式含 instruction/input/output/system")
    parser.add_argument("--vl_json_path", required=True, help="图文 RSVQA json（conversations 含 <|vision_start|>path<|vision_end|>）")
    parser.add_argument("--output_dir", default="./output/Qwen2-VL-2B")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--resized_hw", type=int, default=280)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--f1_threshold", type=float, default=0.6)
    parser.add_argument("--force_explain_in_eval", action="store_true", help="评估时强制要求解释（减少 yes/no）")
    # 训练超参（可以按需改）
    parser.add_argument("--epochs_text", type=float, default=1.0, help="纯文本阶段 epoch")
    parser.add_argument("--epochs_vl", type=float, default=4.0, help="图文阶段 epoch")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--per_device_bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)  # 推荐 2*r
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    # 下载模型（如果没有）
    if not os.path.isdir(args.base_model_dir):
        snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir="./", revision="master")

    tokenizer, processor = build_tokenizer_processor(args.base_model_dir)
    model = build_model(args.base_model_dir)

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)

    # Collator
    collator = MixedVLTextCollator(tokenizer)

    # 纯文本数据集
    text_ds = Dataset.from_json(args.text_json_path)
    process_text_func = make_process_text_func(tokenizer, processor, args.max_length)
    text_dataset = text_ds.map(process_text_func, num_proc=1)

    # 图文数据集：从 vl_json_path 读取后划分 train/test
    with open(args.vl_json_path, "r", encoding="utf-8") as f:
        vl_data = json.load(f)
        if not isinstance(vl_data, list) or len(vl_data) < 20:
            raise ValueError(f"图文数据量过小或格式异常，长度={len(vl_data)}")

    random.seed(42)
    random.shuffle(vl_data)
    split = max(10, int(len(vl_data) * (1 - args.test_ratio)))
    vl_train_data = vl_data[:split]
    vl_test_data = vl_data[split:]

    # 写临时文件（保持你原工作流）
    train_json_path = os.path.join(args.output_dir, "data_vl_train.json")
    test_json_path = os.path.join(args.output_dir, "data_vl_test.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(vl_train_data, f, ensure_ascii=False)
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(vl_test_data, f, ensure_ascii=False)

    vl_train_ds = Dataset.from_json(train_json_path)
    process_vl_func = make_process_vl_func(tokenizer, processor, args.max_length, args.resized_hw)
    vl_train_dataset = vl_train_ds.map(process_vl_func, num_proc=1)

    # SwanLab（建议仅记录训练，评估在主进程再 log）
    swanlab_callback = SwanLabCallback(
        project="Qwen2-VL-RS",
        experiment_name="qwen2-vl-twostage",
        config={
            "base_model": args.base_model_dir,
            "text_json": args.text_json_path,
            "vl_json": args.vl_json_path,
            "vl_train_size": len(vl_train_data),
            "vl_test_size": len(vl_test_data),
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "per_device_bs": args.per_device_bs,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
            "resized_hw": args.resized_hw,
        },
    )

    # 基础 TrainingArguments（两阶段复用，epochs 不同用 replace）
    base_train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        save_on_each_node=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,      # 关键：避免多模态/解码 segfault
        dataloader_pin_memory=False,
        bf16=True,
        report_to="none",
    )

    # --------------------------
    # 阶段 1：纯文本训练
    # --------------------------
    text_args = base_train_args
    text_args.num_train_epochs = args.epochs_text
    text_args.run_name = "stage1_text"

    trainer_text = Trainer(
        model=peft_model,
        args=text_args,
        train_dataset=text_dataset,
        data_collator=collator,
        callbacks=[swanlab_callback],
    )
    trainer_text.train()

    # --------------------------
    # 阶段 2：图文训练（继续同一个 LoRA）
    # --------------------------
    vl_args = base_train_args
    vl_args.num_train_epochs = args.epochs_vl
    vl_args.run_name = "stage2_vl"

    trainer_vl = Trainer(
        model=peft_model,
        args=vl_args,
        train_dataset=vl_train_dataset,
        data_collator=collator,
        callbacks=[swanlab_callback],
    )
    trainer_vl.train()

    # --------------------------
    # 评估：只在主进程做（避免多卡重复评估+重复 log）
    # --------------------------
    if not trainer_vl.is_world_process_zero():
        return

    # 找最后 checkpoint
    last_checkpoint = None
    ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
    if ckpts:
        last_checkpoint = os.path.join(args.output_dir, sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1])
    if not last_checkpoint:
        raise ValueError(f"未找到训练好的 checkpoint，请确认 {args.output_dir} 内有 checkpoint-* 目录")

    print(f"\n[Test] checkpoint = {last_checkpoint}, test_size = {len(vl_test_data)}")

    # base model eval（新建一份 base model，避免被 peft 影响）
    base_model_for_eval = build_model(args.base_model_dir).to("cuda")

    # finetuned model eval：在 base 上加载 LoRA adapter（自动匹配 r/alpha）
    finetuned_model = PeftModel.from_pretrained(base_model_for_eval, model_id=last_checkpoint).to("cuda")

    acc_before, _ = evaluate_vl_dataset(
        base_model_for_eval,
        "Before fine-tune",
        vl_test_data,
        processor,
        max_samples=args.max_eval_samples,
        f1_threshold=args.f1_threshold,
        force_explain=args.force_explain_in_eval,
    )
    acc_after, preds_after = evaluate_vl_dataset(
        finetuned_model,
        "After fine-tune",
        vl_test_data,
        processor,
        max_samples=args.max_eval_samples,
        f1_threshold=args.f1_threshold,
        force_explain=args.force_explain_in_eval,
    )

    swanlab.log({"Acc_Before": acc_before, "Acc_After": acc_after, "Prediction_After": preds_after})
    swanlab.finish()


if __name__ == "__main__":
    main()
