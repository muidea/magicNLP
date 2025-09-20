from flask import Flask, request, jsonify
import jieba
from sentence_transformers import SentenceTransformer, util
import re
import os

app = Flask(__name__)

# ===========================
# 模型加载（可通过环境变量选择）
# ===========================
MODEL_NAME = os.getenv("EMBED_MODEL", "thenlper/gte-small")
LOCAL_FILES_ONLY = bool(os.getenv("LOCAL_FILES_ONLY", "True") == "True")

print(f"Loading embedding model: {MODEL_NAME}, local_files_only={LOCAL_FILES_ONLY}")
model = SentenceTransformer(MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)

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
    return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

def clean_words(words, stopwords):
    return [w for w in words if w not in stopwords and w not in blacklist]

# ===========================
# 关键词提取函数
# ===========================
def extract_keywords(text, top_k=10, business_context=None, sim_threshold=0.3):
    words_zh = [w for w in jieba.cut(text) if len(w) > 1]
    words_en = extract_english_words(text)

    words = clean_words(words_zh, stopwords_zh).copy()
    words += clean_words(words_en, stopwords_en)
    words = list(set(words))

    if not words:
        return []

    text_emb = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    word_embs = model.encode(words, convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(text_emb, word_embs)[0].cpu().numpy()
    word_score_pairs = list(zip(words, cos_scores))

    if not business_context:
        word_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in word_score_pairs[:top_k]]

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
# API 接口统一前缀 + 服务名
# ===========================
API_PREFIX = "/api/v1/nlp_service"

# ---------------------------
# 单条文本 embedding
# ---------------------------
@app.route(f"{API_PREFIX}/embedding/single", methods=["POST"])
def embed_single():
    """
    单条文本向量生成接口
    请求参数:
    - text (str, 必选): 待生成向量的文本内容

    返回示例:
    {
        "vector": [浮点数向量列表]
    }
    """
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400
    vector = model.encode([text], normalize_embeddings=True)[0].tolist()
    return jsonify({"vector": vector})

# ---------------------------
# 批量文本 embedding
# ---------------------------
@app.route(f"{API_PREFIX}/embedding/batch", methods=["POST"])
def embed_batch():
    """
    批量文本向量生成接口
    请求参数:
    - texts (list[str], 必选): 待生成向量的文本列表，至少包含一条文本

    返回示例:
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
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    return jsonify({"vectors": vectors})

# ---------------------------
# 中文+英文关键词提取
# ---------------------------
@app.route(f"{API_PREFIX}/keywords/extract", methods=["POST"])
def keywords_api():
    """
    中文+英文关键词提取接口
    请求参数:
    - text (str, 必选): 待提取关键词的文本
    - top_k (int, 可选, 默认10): 返回关键词数量
    - business_context (str, 可选): 业务语境，用于语义过滤关键词
    - threshold (float, 可选, 默认0.3): 业务语境相似度阈值，范围 0~1

    返回示例:
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

# ---------------------------
# 健康检查接口
# ---------------------------
@app.route(f"{API_PREFIX}/health", methods=["GET"])
def health_check():
    """
    服务健康检查接口
    请求参数: 无
    返回示例:
    {
        "status": "ok",
        "model": "BAAI/bge-small-zh-v1.5"
    }
    """
    return jsonify({"status": "ok", "model": MODEL_NAME})

# ===========================
# 启动服务
# ===========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
