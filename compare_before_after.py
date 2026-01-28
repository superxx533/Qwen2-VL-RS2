import json

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType

from qwen_vl_utils import process_vision_info


def build_messages_from_conversation(user_value: str):
    """
    从 data_vl_*.json 里的 user 字段解析出：问题文本 + 图像路径
    格式约定：
        \"问题文本 <|vision_start|>图像路径<|vision_end|>\"
    如果没有问题文本（老的 COCO 格式），则使用一个默认问题。
    """
    if "<|vision_start|>" in user_value and "<|vision_end|>" in user_value:
        parts = user_value.split("<|vision_start|>")
        question_text = parts[0].strip()
        image_path = parts[1].split("<|vision_end|>")[0]
    else:
        # 兼容老格式：只包含 COCO Yes: + 图像路径
        question_text = "请描述这张遥感图像中的场景和目标。"
        image_path = user_value.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": question_text,
                },
            ],
        }
    ]
    return messages, question_text, image_path


def predict(messages, model, processor):
    """给定 messages 和模型，生成一个文本回答。"""
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
    # 将输入移动到与模型相同的设备，避免 CPU/GPU 混用错误
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs[0]


def main():
    # ==== 1. 加载对比用的数据 ====
    # 这里默认使用 data_vl_test.json（由 train.py 基于 RSVQA 生成）
    test_json_path = "data_vl_test.json"
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"加载测试样本数量: {len(test_data)}")

    # ==== 2. 加载原始模型（未微调）和 LoRA 微调后的模型 ====
    base_model_path = "/home/lx/2026/Qwen2-VL-RS/Qwen/Qwen2-VL-2B-Instruct"
    lora_checkpoint_path = "/home/lx/2026/Qwen2-VL-RS/output/Qwen2-VL-2B/checkpoint-672"

    print("加载原始 Qwen2-VL-2B-Instruct 模型 (Before fine-tuning)...")
    base_model_before = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    print("加载 LoRA 适配后的模型 (After fine-tuning)...")
    # 为了避免在同一个实例上原地加载 LoRA，这里重新加载一份基座模型
    base_model_for_lora = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    lora_model = PeftModel.from_pretrained(
        base_model_for_lora,
        model_id=lora_checkpoint_path,
        config=lora_config,
    )

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # ==== 3. 对每条样本分别用“训练前/训练后”推理，并打印对比 ====
    max_samples = min(20, len(test_data))  # 避免一次性跑太多
    print(f"\n开始对比前 {max_samples} 条样本：\n")

    for idx, item in enumerate(test_data[:max_samples], start=1):
        user_value = item["conversations"][0]["value"]
        gt_answer = item["conversations"][1]["value"]

        messages, question_text, image_path = build_messages_from_conversation(user_value)

        print("=" * 80)
        print(f"样本 #{idx}")
        print(f"图像路径: {image_path}")
        print(f"问题(Q): {question_text}")
        print(f"真实答案(GT): {gt_answer}")

        # 原始模型回答（未微调）
        try:
            ans_before = predict(messages, base_model_before, processor)
        except Exception as e:
            ans_before = f"[推理失败: {e}]"

        # 微调后模型回答
        try:
            ans_after = predict(messages, lora_model, processor)
        except Exception as e:
            ans_after = f"[推理失败: {e}]"

        print(f"\n原始模型回答 (Before fine-tuning): {ans_before}")
        print(f"微调后模型回答 (After fine-tuning):  {ans_after}")

    print("\n对比完成。")


if __name__ == "__main__":
    main()

