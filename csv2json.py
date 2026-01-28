import pandas as pd
import json

# 载入RSVQA CSV文件
csv_path = './rsvqa-dataset.csv'
df = pd.read_csv(csv_path, encoding='utf-8')

conversations = []

# 添加对话数据（遥感视觉问答格式）
for i in range(len(df)):
    image_path = df.iloc[i]['image_path']
    question = df.iloc[i]['question']
    answer = df.iloc[i]['answer']
    
    # 构建对话格式：用户提供图像和问题，助手回答
    conversations.append({
        "id": f"rsvqa_{i+1}",
        "conversations": [
            {
                "from": "user",
                "value": f"{question} <|vision_start|>{image_path}<|vision_end|>"
            },
            {
                "from": "assistant",
                "value": answer
            }
        ]
    })

# 保存为JSON文件
output_path = 'data_vl.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

print(f'数据转换完成！共转换了{len(conversations)}条对话数据')
print(f'输出文件: {output_path}')
print(f'\n数据预览（前3条）:')
for i, conv in enumerate(conversations[:3]):
    print(f"\n对话 {i+1}:")
    print(f"  用户: {conv['conversations'][0]['value'][:100]}...")
    print(f"  助手: {conv['conversations'][1]['value']}")