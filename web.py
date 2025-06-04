from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import json
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import requests

app = Flask(__name__)
ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')
HISTORY_FILE = "history.json"
CONFIG_FILE = "config.json"

# 默认配置
model_map = {
    "DeepSeek (支持流式输出)": "https://api.deepseek.com/v1/chat/completions",
    "Qwen (通义千问)": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
    "Ollama 本地": "http://localhost:11434/api/chat/completions"
}

config = {"model": "DeepSeek", "url": model_map["DeepSeek (支持流式输出)"], "key": ""}
if os.path.exists(CONFIG_FILE):
    try:
        config.update(json.load(open(CONFIG_FILE, 'r', encoding='utf-8')))
    except: pass

@app.route('/history/import', methods=['POST'])
def import_history():
    data = request.get_json()
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True})

@app.route("/")
def index():
    return render_template("index.html", 
        model_names=list(model_map.keys()), 
        config=config,
        model_map=model_map  # 关键在这里
    )

@app.route('/ocr', methods=['POST'])
def ocr_image():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = ocr_model.ocr(img, cls=True)
    text = "\n".join([line[1][0] for line in result[0]])
    return jsonify({"text": text})

@app.route('/ask', methods=['POST'])
def ask():
    q = request.json.get('question', '')
    model = config['model']
    api_url = config['url']
    api_key = config['key']
    headers = {"Content-Type": "application/json"}
    data = {}
    # DeepSeek
    if model == "DeepSeek":
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一名图论专家，给出解题过程(不超200字)"},
                {"role": "user", "content": q}
            ],
            "temperature": 0.5,
            "max_tokens": 512
        }
    # Qwen
    elif model == "Qwen (通义千问)":
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = {
            "model": "qwen-turbo",
            "input": {
                "messages": [
                    {"role": "system", "content": "你是一名图论专家，给出解题过程(不超200字)"},
                    {"role": "user", "content": q}
                ]
            }
        }
    # Ollama
    elif model == "Ollama 本地":
        data = {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": "你是一名图论专家，给出解题过程(不超200字)"},
                {"role": "user", "content": q}
            ]
        }
    else:
        return jsonify({"error": "未知模型"}), 400

    try:
        r = requests.post(api_url, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        # DeepSeek/Ollama
        if model in ["DeepSeek", "Ollama 本地"]:
            a = r.json()["choices"][0]["message"]["content"]
        # Qwen
        elif model == "Qwen (通义千问)":
            a = r.json()["output"]["choices"][0]["message"]["content"]
        else:
            a = "未知模型"
        save_to_history(q, a)
        return jsonify({"answer": a})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask/stream', methods=['POST'])
def ask_stream():
    q = request.json.get('question', '')
    model = config['model']
    api_url = config['url']
    api_key = config['key']
    headers = {"Content-Type": "application/json"}

    if model != "DeepSeek (支持流式输出)":
        return jsonify({"error": "仅 DeepSeek 支持流式输出"}), 400

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一名图论专家，给出解题过程(不超200字)"},
            {"role": "user", "content": q}
        ],
        "temperature": 0.5,
        "max_tokens": 512,
        "stream": True
    }

    def generate():
        with requests.post(api_url, headers=headers, json=data, stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith('data:'):
                    try:
                        chunk = json.loads(line[5:].strip())
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except Exception:
                        continue

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/config', methods=['POST'])
def update_config():
    config['model'] = request.form.get('model')
    config['url'] = request.form.get('url')
    config['key'] = request.form.get('key')
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    return jsonify({"success": True})

@app.route('/history', methods=['GET'])
def get_history():
    if os.path.exists(HISTORY_FILE):
        return jsonify(json.load(open(HISTORY_FILE, 'r', encoding='utf-8')))
    return jsonify([])

@app.route('/history/delete', methods=['POST'])
def delete_history():
    index = request.json.get("index")
    if os.path.exists(HISTORY_FILE):
        data = json.load(open(HISTORY_FILE, 'r', encoding='utf-8'))
        if 0 <= index < len(data):
            data.pop(index)
            json.dump(data, open(HISTORY_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return jsonify({"success": True})

def save_to_history(q, a):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            history = json.load(open(HISTORY_FILE, 'r', encoding='utf-8'))
        except: pass
    history.append((q, a))
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5001, debug=True)