import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import os
import random
import re


def process_func(example):
    """
    将RSVQA数据集进行预处理
    处理格式：问题文本 + 图像路径
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    # 从输入中提取图像路径和问题文本
    # 格式: "问题文本 <|vision_start|>图像路径<|vision_end|>"
    if "<|vision_start|>" in input_content and "<|vision_end|>" in input_content:
        parts = input_content.split("<|vision_start|>")
        question_text = parts[0].strip()  # 问题文本
        file_path = parts[1].split("<|vision_end|>")[0]  # 图像路径
    else:
        # 兼容旧格式（如果还有的话）
        file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        question_text = "请描述这张遥感图像。"  # 默认问题

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{file_path}",
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": question_text},  # 使用问题文本而不是固定的"COCO Yes:"
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # 将输入移动到与模型相同的设备上
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# ==================== 基础配置 ====================
BASE_MODEL_DIR = "./Qwen/Qwen2-VL-2B-Instruct"
DATA_JSON_PATH = "data_vl.json"  # 由 csv2json.py 生成
TRAIN_JSON_PATH = "data_vl_train.json"
TEST_JSON_PATH = "data_vl_test.json"
OUTPUT_DIR = "./output/Qwen2-VL-2B"
CHECKPOINT_NAME = None  # 训练完成后自动取最后一个 epoch 的 checkpoint

TEST_RATIO = 0.1  # 按 9:1 划分验证/测试
MAX_EVAL_SAMPLES = 200  # 评估最多抽取多少条（避免生成太耗时）

# 在modelscope上下载Qwen2-VL模型到本地目录下（若目录已存在则复用）
if not os.path.isdir(BASE_MODEL_DIR):
    snapshot_download("Qwen/Qwen2-VL-2B-Instruct", cache_dir="./", revision="master")

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取 json 文件，简单划分 train/test
with open(DATA_JSON_PATH, 'r', encoding="utf-8") as f:
    data = json.load(f)
    if not isinstance(data, list) or len(data) < 20:
        raise ValueError(f"data_vl.json 数据量过小或格式异常，长度={len(data)}")

# 打乱并按比例划分 train / test
random.seed(42)
random.shuffle(data)
split = max(10, int(len(data) * (1 - TEST_RATIO)))
train_data = data[:split]
test_data = data[split:]

with open(TRAIN_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False)

with open(TEST_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False)

train_ds = Dataset.from_json(TRAIN_JSON_PATH)
train_dataset = train_ds.map(process_func)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=4,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-RS",
    experiment_name="qwen2-vl-rsvqa",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct",
        "dataset": "RSVQA (Remote Sensing Visual Question Answering)",
        "github": "https://github.com/syvlo/RSVQA",
        "task": "Remote Sensing VQA",
        "train_data_number": len(train_data),
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

# 开启模型训练
trainer.train()

# ====================测试模式===================
# 配置测试参数
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
    bias="none",
)

# 获取测试模型：取最后一个 epoch 的 checkpoint
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    # 形如 checkpoint-1, checkpoint-2 ...
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if ckpts:
        last_checkpoint = os.path.join(OUTPUT_DIR, sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1])

if not last_checkpoint:
    raise ValueError(f"未找到训练好的checkpoint，请确认 {OUTPUT_DIR} 内有 checkpoint-* 目录")

val_peft_model = PeftModel.from_pretrained(model, model_id=last_checkpoint, config=val_config)

# 读取测试数据
with open(TEST_JSON_PATH, "r", encoding="utf-8") as f:
    test_dataset = json.load(f)


def _normalize_answer(text: str):
    t = text.strip().lower()
    # 抽取第一个整数用于计数题对比；若不存在，保留原文
    m = re.search(r"-?\d+", t)
    if m:
        return m.group(0)
    return t


def evaluate_model(model_to_use, label: str, dataset, max_samples=MAX_EVAL_SAMPLES):
    total = 0
    correct = 0
    preds_for_log = []
    for item in dataset[:max_samples]:
        input_image_prompt = item["conversations"][0]["value"]
        expected_answer = item["conversations"][1]["value"]

        if "<|vision_start|>" in input_image_prompt and "<|vision_end|>" in input_image_prompt:
            parts = input_image_prompt.split("<|vision_start|>")
            question_text = parts[0].strip()
            origin_image_path = parts[1].split("<|vision_end|>")[0]
        else:
            origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
            question_text = "请描述这张遥感图像。"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": origin_image_path},
                {"type": "text", "text": question_text},
            ]}]

        pred = predict(messages, model_to_use)
        total += 1
        if _normalize_answer(pred) == _normalize_answer(expected_answer):
            correct += 1
        if total <= 5:  # 只打印少量样例，避免刷屏
            print(f"\n[{label}] Q: {question_text}")
            print(f"GT: {expected_answer}")
            print(f"P : {pred}")
        preds_for_log.append(swanlab.Image(origin_image_path, caption=f"{label}: {pred}"))
    acc = correct / total if total else 0.0
    print(f"\n[{label}] Accuracy (exact/normalized): {correct}/{total} = {acc:.2%}")
    return acc, preds_for_log


# 评估：原始模型 vs 微调模型
print(f"\n使用测试集大小: {len(test_dataset)}, checkpoint: {last_checkpoint}")
base_model_for_eval = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

acc_before, _ = evaluate_model(base_model_for_eval, "Before fine-tune", test_dataset)
val_peft_model = val_peft_model.to("cuda")
acc_after, preds_after = evaluate_model(val_peft_model, "After fine-tune", test_dataset)

# 仅记录微调后的预测图片到 swanlab，避免重复占空间
swanlab.log({"Prediction_After": preds_after, "Acc_Before": acc_before, "Acc_After": acc_after})

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()