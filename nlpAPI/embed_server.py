from flask import Flask, request, jsonify
import jieba
from sentence_transformers import SentenceTransformer, util
import base64
import re
import os
import time

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
# Embedding 公共函数
# ===========================
def json_payload():
    return request.get_json(silent=True) or {}

def error_response(message, status_code=400, param=None, error_type="invalid_request_error"):
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None
        }
    }), status_code

def parse_text_input(value, field_name="input"):
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        return [value]

    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        if not all(isinstance(item, str) for item in value):
            raise TypeError(f"{field_name} must be a string or an array of strings")
        if any(item == "" for item in value):
            raise ValueError(f"{field_name} contains empty text")
        return value

    raise TypeError(f"{field_name} must be a string or an array of strings")

def encode_texts(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()

def count_prompt_tokens(texts):
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenized = tokenizer(texts, add_special_tokens=True)
            return sum(len(input_ids) for input_ids in tokenized.get("input_ids", []))
        except Exception:
            pass

    return sum(len(text) for text in texts)

def encode_vector_base64(vector):
    import numpy as np

    raw = np.asarray(vector, dtype="float32").tobytes()
    return base64.b64encode(raw).decode("ascii")

def build_openai_embeddings_response(texts, requested_model=None, encoding_format="float"):
    if encoding_format not in ("float", "base64"):
        raise ValueError("encoding_format must be 'float' or 'base64'")

    vectors = encode_texts(texts)
    data = []
    for index, vector in enumerate(vectors):
        embedding = encode_vector_base64(vector) if encoding_format == "base64" else vector
        data.append({
            "object": "embedding",
            "embedding": embedding,
            "index": index
        })

    prompt_tokens = count_prompt_tokens(texts)
    return {
        "object": "list",
        "data": data,
        "model": requested_model or MODEL_NAME,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    }

def build_ollama_embed_response(texts, requested_model=None, started_at=None):
    started_at = started_at or time.perf_counter()
    vectors = encode_texts(texts)
    duration_ns = int((time.perf_counter() - started_at) * 1_000_000_000)

    return {
        "model": requested_model or MODEL_NAME,
        "embeddings": vectors,
        "total_duration": duration_ns,
        "load_duration": 0,
        "prompt_eval_count": count_prompt_tokens(texts)
    }

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
    data = json_payload()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text"}), 400
    vector = encode_texts([text])[0]
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
    data = json_payload()
    texts = data.get("texts", [])
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Invalid texts"}), 400
    vectors = encode_texts(texts)
    return jsonify({"vectors": vectors})

# ---------------------------
# OpenAI 兼容 embeddings
# ---------------------------
@app.route("/v1/embeddings", methods=["POST"])
@app.route("/api/v1/embeddings", methods=["POST"])
@app.route(f"{API_PREFIX}/embeddings", methods=["POST"])
def openai_embeddings_api():
    """
    OpenAI 兼容向量生成接口
    请求格式:
    {
        "model": "模型名，可选",
        "input": "文本" 或 ["文本1", "文本2"],
        "encoding_format": "float" 或 "base64，可选，默认 float"
    }
    响应格式:
    {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [...], "index": 0}
        ],
        "model": "模型名",
        "usage": {"prompt_tokens": 10, "total_tokens": 10}
    }
    """
    data = json_payload()
    try:
        texts = parse_text_input(data.get("input"), "input")
        encoding_format = data.get("encoding_format", "float")
        response = build_openai_embeddings_response(
            texts,
            requested_model=data.get("model"),
            encoding_format=encoding_format
        )
        return jsonify(response)
    except (TypeError, ValueError) as e:
        return error_response(str(e), 400, "input")

# ---------------------------
# Ollama 兼容 embed
# ---------------------------
@app.route("/api/embed", methods=["POST"])
@app.route(f"{API_PREFIX}/embed", methods=["POST"])
def ollama_embed_api():
    """
    Ollama 兼容向量生成接口
    请求格式:
    {
        "model": "模型名，可选",
        "input": "文本" 或 ["文本1", "文本2"]
    }
    响应格式:
    {
        "model": "模型名",
        "embeddings": [[...]],
        "total_duration": 123,
        "load_duration": 0,
        "prompt_eval_count": 10
    }
    """
    started_at = time.perf_counter()
    data = json_payload()
    try:
        texts = parse_text_input(data.get("input"), "input")
        return jsonify(build_ollama_embed_response(
            texts,
            requested_model=data.get("model"),
            started_at=started_at
        ))
    except (TypeError, ValueError) as e:
        return error_response(str(e), 400, "input")

# ---------------------------
# Ollama 旧版 embeddings
# ---------------------------
@app.route("/api/embeddings", methods=["POST"])
@app.route(f"{API_PREFIX}/embedding", methods=["POST"])
def ollama_legacy_embeddings_api():
    """
    Ollama 旧版单条向量接口
    请求格式:
    {
        "model": "模型名，可选",
        "prompt": "文本"
    }
    响应格式:
    {
        "embedding": [...]
    }
    """
    data = json_payload()
    prompt = data.get("prompt", data.get("input", ""))
    try:
        text = parse_text_input(prompt, "prompt")
        if len(text) != 1:
            return error_response("prompt must be a single string", 400, "prompt")
        vector = encode_texts(text)[0]
        return jsonify({"embedding": vector})
    except (TypeError, ValueError) as e:
        return error_response(str(e), 400, "prompt")

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
