# 导入所需的库
import os
import pandas as pd
import json
from PIL import Image

MAX_DATA_NUMBER = 1500

# RSVQA数据集路径配置
# 假设RSVQA数据集已经下载，包含以下结构：
# - rsvqa_images/ 目录：包含遥感图像文件（.tif 等）
# - rsvqa_questions.json：包含问题列表（带 img_id / answers_ids）
# - rsvqa_answers.json：包含答案列表（带 question_id / answer）
RSVQA_IMAGES_DIR = 'rsvqa_images'          # RSVQA图像目录
RSVQA_QUESTIONS_PATH = 'rsvqa_questions.json'  # RSVQA问题JSON文件路径
RSVQA_ANSWERS_PATH = 'rsvqa_answers.json'      # RSVQA答案JSON文件路径


def load_rsvqa_data(questions_path, answers_path, images_dir, max_samples=500):
    """
    加载RSVQA数据集

    rsvqa_questions.json 示例：
        {
          "questions": [
            {
              "id": 0,
              "img_id": 0,
              "question": "Is it a rural or an urban area",
              "answers_ids": [0]
            },
            ...
          ]
        }

    rsvqa_answers.json 示例：
        {
          "answers": [
            {
              "id": 0,
              "question_id": 0,
              "answer": "urban"
            },
            ...
          ]
        }
    """
    # 检查JSON文件是否存在
    if not os.path.exists(questions_path):
        print(f'错误：找不到RSVQA问题JSON文件: {questions_path}')
        return None
    if not os.path.exists(answers_path):
        print(f'错误：找不到RSVQA答案JSON文件: {answers_path}')
        return None

    # 检查图像目录是否存在
    if not os.path.exists(images_dir):
        print(f'警告：图像目录不存在: {images_dir}')
        print('请确保RSVQA图像文件已下载并放在正确的目录中')
        return None

    # 读取JSON文件
    with open(questions_path, 'r', encoding='utf-8') as f_q:
        q_data = json.load(f_q)
    with open(answers_path, 'r', encoding='utf-8') as f_a:
        a_data = json.load(f_a)

    # 从外层字典中取出数组
    questions_list = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
    answers_list = a_data.get("answers", a_data) if isinstance(a_data, dict) else a_data

    # 建立 question_id -> answer 文本 映射（这里只取第一个答案）
    answer_by_question_id = {}
    for ans in answers_list:
        qid = ans.get("question_id")
        ans_text = ans.get("answer")
        if qid is None or ans_text is None:
            continue
        # 如果一个问题有多个答案，只保留第一个即可
        if qid not in answer_by_question_id:
            answer_by_question_id[qid] = ans_text

    # 初始化存储列表
    image_paths = []
    questions = []
    answers = []

    # 处理数据
    if not isinstance(questions_list, list):
        print('错误：questions 数据格式不是列表，请检查 rsvqa_questions.json')
        return None

    total = min(max_samples, len(questions_list))
    print(f'共找到 {len(questions_list)} 条问题记录，本次最多处理 {total} 条')

    for i, item in enumerate(questions_list[:total]):
        q_id = item.get("id")
        img_id = item.get("img_id")
        question_text = item.get("question", "")

        if q_id is None or img_id is None or not question_text:
            print(f'警告：第{i + 1}条问题缺少 id/img_id/question，跳过')
            continue

        # 取出该问题对应的答案（只用一个答案）
        # questions 里也有 answers_ids，但我们直接用 answers.json 里的映射更稳妥
        answer_text = answer_by_question_id.get(q_id)
        if not answer_text:
            print(f'警告：找不到问题ID {q_id} 的答案，跳过')
            continue

        # RSVQA 图像命名一般为 \"{img_id}.tif\"
        image_id_str = str(img_id)
        image_file = None
        for ext in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(images_dir, f'{image_id_str}{ext}')
            if os.path.exists(potential_path):
                image_file = potential_path
                break

        if not image_file:
            print(f'警告：找不到图像文件(按 img_id 搜索): {img_id}，跳过')
            continue

        # 转换为绝对路径
        image_path = os.path.abspath(image_file)
        image_paths.append(image_path)
        questions.append(question_text)
        answers.append(answer_text)

        # 每处理50条数据打印一次进度
        if (len(questions)) % 50 == 0:
            print(f'处理进度: {len(questions)}/{total} ({len(questions) / total * 100:.1f}%)')

    # 创建DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'question': questions,
        'answer': answers
    })

    print(f'有效样本数: {len(df)}')
    return df

# 主程序
if __name__ == '__main__':
    print('开始处理RSVQA数据集...')
    
    # 加载RSVQA数据
    df = load_rsvqa_data(RSVQA_QUESTIONS_PATH, RSVQA_ANSWERS_PATH, RSVQA_IMAGES_DIR, MAX_DATA_NUMBER)
    
    if df is not None and len(df) > 0:
        # 保存为CSV文件
        csv_path = './rsvqa-dataset.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f'数据处理完成！共处理了{len(df)}条数据')
        print(f'数据已保存到: {csv_path}')
        print(f'\n数据预览:')
        print(df.head())
    else:
        print('数据处理失败，请检查数据集路径和格式')