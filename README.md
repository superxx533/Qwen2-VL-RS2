## Qwen2-VL-RS：基于 Qwen2-VL 的遥感视觉问答（RSVQA）微调项目

本项目基于阿里巴巴的多模态大模型 Qwen2-VL-2B-Instruct，将其从通用图文理解任务微调为 遥感视觉问答（Remote Sensing VQA, RSVQA） 任务。

输入为：遥感图像 + 问题文本，输出为：简洁的答案（如 yes/no 或数字）。

------

### 目录结构

Qwen2-VL-RS/

├── load.py          # 从 ModelScope加载 Qwen2-VL 基座模型到本地 Qwen/ 目录

├── data2csv.py       # 从 RSVQA 原始 JSON + 图像生成 CSV

├── csv2json.py       # 从 CSV 生成对话格式 data_vl.json

├── train.py         # 使用 LoRA 对 Qwen2-VL 进行微调 + 自动评估

├── compare_before_after.py # 对比微调前/后的回答（Before vs After）

├── README.md     #README文件

├── rsvqa_images/      # RSVQA 遥感图像（.tif）

├── rsvqa_questions.json   # RSVQA 问题 JSON（questions 数组）

├── rsvqa_answers.json    # RSVQA 答案 JSON（answers 数组）

├── data_vl.json       # 训练用对话数据（由 csv2json.py 生成）

├── data_vl_train.json    # 训练集（train.py 自动生成）

├── data_vl_test.json    # 测试集（train.py 自动生成）

├── output/Qwen2-VL-2B/   # LoRA 训练输出（checkpoint-*）

└── Qwen/Qwen2-VL-2B-Instruct/ # Qwen2-VL 基座模型本地权重



------

### 环境依赖

- Python 3.10+

- PyTorch（支持 CUDA）

- Transformers

- PEFT

- datasets

- modelscope

- Pillow

- swanlab

- qwen-vl-utils（随 Qwen2-VL 提供）

示例安装：

pip install requirements.txt

------

### RSVQA 数据准备

1.下载 RSVQA 数据集（官方 GitHub 或 Zenodo）。

2.解压图像到项目根目录下的 rsvqa_images/，图像名称一般为 0.tif, 1.tif, ...。

3.将官方提供的 rsvqa_questions.json、rsvqa_answers.json 放在项目根目录。

数据格式（简要）：

- rsvqa_questions.json：

```
{

 "questions": [

  {

   "id": 0,

   "img_id": 0,

   "question": "Is it a rural or an urban area",

   "answers_ids": [0]

  }

 ]

}
```



- rsvqa_answers.json：

```
{

 "answers": [

  {

   "id": 0,

   "question_id": 0,

   "answer": "urban"

  }

 ]

}
```



------

### 脚本说明

#### 1. load.py：加载/下载 Qwen2-VL 基座模型

(1)功能：

- 从 ModelScope下载 Qwen/Qwen2-VL-2B-Instruct；

- 将模型权重和配置保存到本地 Qwen/Qwen2-VL-2B-Instruct/ 目录，供后续 train.py、compare_before_after.py、推理脚本复用。

(2)使用：

python load.py

#### 2. data2csv.py：RSVQA → CSV

（1）功能：

- 读取 rsvqa_questions.json + rsvqa_answers.json，通过 question_id 对齐问题和答案；

- 按 img_id 匹配 rsvqa_images/ 中的图像；

- 生成 rsvqa-dataset.csv，包含三列：

- image_path：图像绝对路径

- question：问题文本

- answer：答案（yes/no 或数字等）

（2）使用：

python data2csv.py

#### 3. csv2json.py：CSV → 对话 JSON

（1）功能：

- 读取 rsvqa-dataset.csv；

- 生成对话格式的 data_vl.json，每条数据结构如下：

```
{

 "id": "rsvqa_1",

 "conversations": [

  {

   "from": "user",

   "value": "QUESTION_TEXT <|vision_start|>/abs/path/to/image.tif<|vision_end|>"

  },

  {

   "from": "assistant",

   "value": "ANSWER_TEXT"

  }

 ]

}
```

（2）使用：

python csv2json.py

#### 4. train.py：LoRA 微调 + 自动评估

（1）功能：

- 从 data_vl.json 随机划分训练集 / 测试集（默认 9:1），保存为 data_vl_train.json、data_vl_test.json；

- 使用 Qwen2-VL-2B-Instruct 作为基座模型，配置 LoRA（rank=64 等）进行微调；

- 使用 HuggingFace Trainer 完成训练，checkpoint 保存在 output/Qwen2-VL-2B/checkpoint-*；

- 训练结束后，自动：加载最新 checkpoint；在测试集上评估“微调前 / 微调后”准确率（使用数字/yes/no 归一化比较）；打印若干样例；将结果记录到 SwanLab。

（2）使用：

python train.py

#### 5. compare_before_after.py：逐样本对比 Before vs After

（1）功能：

- 从 data_vl_test.json 读取测试样本；

- 构造 messages（图像 + 问题文本）；

- 分别使用：基座模型（未加载 LoRA）：base_model_before以及加载了 LoRA checkpoint 的模型：lora_model，对每条样本打印：图像路径 / 问题 / 真实答案（GT）、原始模型回答（Before fine-tuning）、微调后模型回答（After fine-tuning）

（2）使用前请在脚本中检查：

- base_model_path 指向你的 Qwen2-VL 本地目录；

- lora_checkpoint_path 指向实际存在的 output/Qwen2-VL-2B/checkpoint-XXX 目录。

（3）使用：

python compare_before_after.py

------

### 推荐使用流程

1. 下载基座模型

```
python load.py    #从 ModelScope 下载模型到本地
```


2. 准备数据

```
#准备 RSVQA 图像和 JSON

#rsvqa_images/、rsvqa_questions.json、rsvqa_answers.json
```

3.数据预处理

```
python data2csv.py   *# 生成 rsvqa-dataset.csv*

python csv2json.py   *# 生成 data_vl.json
```

4.训练与评估

```
python train.py     *# 训练 LoRA，并在测试集上评估 Before/After 准确率*
```

5.前后对比（可选，更详细）

```
python compare_before_after.py
```

------

### 注意事项

- 显存：Qwen2-VL-2B + LoRA 在 12GB 以上显卡上较为合适，如显存不足，可减小 per_device_train_batch_size 或提高 gradient_accumulation_steps。

- 计数题难度：对复杂遥感场景的精确计数（几十 / 几百目标）本身就很难，即使微调后仍可能出错；本项目重点在于让模型学会按 RSVQA 的标签形式回答（短 yes/no/数字）。

- 路径一致性：保证训练脚本、对比脚本中使用的 base_model_path、lora_checkpoint_path 与实际文件路径一致。

