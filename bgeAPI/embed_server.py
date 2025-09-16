# merged_server_multilang.py
from flask import Flask, request, jsonify
import jieba
from sentence_transformers import SentenceTransformer, util
import re
import os
import numpy as np

app = Flask(__name__)

# 加载本地模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

# ===========================
# 停用词 & 黑名单加载
# ===========================
def load_wordlist(filepath):
    words = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    words.add(word.lower())
    except FileNotFoundError:
        print(f"{filepath} 文件未找到")
    return words

stopwords_zh = load_wordlist("stopwords_zh.txt")
stopwords_en = load_wordlist("stopwords_en.txt")
blacklist = load_wordlist("blacklist.txt")

# ===========================
# 文本预处理
# ===========================
def extract_english_words(text):
    """提取英文单词"""
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

def clean_words(words, stopwords):
    """去掉停用词和黑名单"""
    return [w for w in words if w not in stopwords and w not in blacklist]

# ===========================
# 关键词提取函数
# ===========================
def extract_keywords(text, top_k=10, business_context=None, sim_threshold=0.3):
    # 分词
    words_zh = [w for w in jieba.cut(text) if len(w) > 1]
    words_en = extract_english_words(text)

    # 去停用词和黑名单
    words = clean_words(words_zh, stopwords_zh).copy()
    words += clean_words(words_en, stopwords_en)
    words = list(set(words))

    if not words:
        return []

    # 基于语义相似度的关键词排序
    text_emb = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    word_embs = model.encode(words, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(text_emb, word_embs)[0].cpu().numpy()
    word_score_pairs = list(zip(words, cos_scores))

    # 如果没有业务语境，直接取前 top_k
    if not business_context:
        word_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in word_score_pairs[:top_k]]

    # 有业务语境时，进一步相似度过滤
    biz_emb = model.encode([business_context], convert_to_tensor=True, normalize_embeddings=True)[0]
    filtered_pairs = []
    for word, score in word_score_pairs:
        w_emb = model.encode([word], convert_to_tensor=True, normalize_embeddings=True)[0]
        sim = util.cos_sim(w_emb, biz_emb).item()
        if sim >= sim_threshold:
            filtered_pairs.append((word, score))

    filtered_pairs.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in filtered_pairs[:top_k]]

# ===========================
# 单条文本 embedding 接口
# ===========================
@app.route("/bge/embed", methods=["POST"])
def embed():
    """
    单条文本向量生成接口
    请求格式:
    {
        "text": "字符串文本"
    }
    响应格式:
    {
        "vector": [浮点数向量列表]
    }
    """
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400

    vector = model.encode([text])[0].tolist()
    return jsonify({"vector": vector})

# ===========================
# 批量文本 embedding 接口
# ===========================
@app.route("/bge/embed_bulk", methods=["POST"])
def embed_bulk():
    """
    批量文本向量生成接口
    请求格式:
    {
        "texts": ["文本1", "文本2", ...]
    }
    响应格式:
    {
        "vectors": [
            [向量1], [向量2], ...
        ]
    }
    """
    data = request.json
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Invalid texts"}), 400

    vectors = model.encode(texts).tolist()
    return jsonify({"vectors": vectors})

# ===========================
# 中文+英文关键词提取接口
# ===========================
@app.route("/bge/keywords", methods=["POST"])
def keywords_api():
    """
    中文+英文关键词提取接口
    请求格式:
    {
        "text": "字符串文本",
        "top_k": 关键词数量（整数，可选，默认10）,
        "business_context": "业务语境字符串，可选",
        "threshold": 相似度阈值（浮点数，可选，默认0.3）
    }
    响应格式:
    {
        "keywords": ["关键词1", "关键词2", ...]
    }
    """
    try:
        data = request.json
        text = data.get("text", "").strip()
        top_k = int(data.get("top_k", 10))
        business_context = data.get("business_context", None)
        sim_threshold = float(data.get("threshold", 0.3))

        if not text:
            return jsonify({"error": "text 参数不能为空"}), 400

        keywords = extract_keywords(
            text, top_k=top_k,
            business_context=business_context,
            sim_threshold=sim_threshold
        )
        return jsonify({"keywords": keywords})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# 启动服务
# ===========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
