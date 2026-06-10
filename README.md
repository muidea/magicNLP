# magicNLP

magicNLP 是一个轻量级 NLP HTTP 服务，基于 `sentence-transformers` 提供文本向量生成和中英文关键词提取能力。服务默认面向小型知识库/RAG 场景，使用 `intfloat/multilingual-e5-small`，并支持 E5 模型推荐的 `query:` / `passage:` 前缀约定。

## 功能

- OpenAI 兼容 embeddings 接口：`POST /v1/embeddings`
- Ollama 兼容 embed 接口：`POST /api/embed`
- 历史兼容接口：`/api/v1/nlp_service/embedding/single`、`/api/v1/nlp_service/embedding/batch`
- 中英文关键词提取：`POST /api/v1/nlp_service/keywords/extract`
- 健康检查：`GET /health`、`GET /api/v1/nlp_service/health`

## 目录

```text
.
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── cache/
│   └── huggingface/
├── nlpAPI/
│   ├── embed_server.py
│   ├── stopwords_zh.txt
│   ├── stopwords_en.txt
│   └── blacklist.txt
└── nlpDemo/
    └── main.go
```

## 快速开始：Docker Compose

首次启动会自动下载模型到项目本地目录 `cache/huggingface/`，后续会复用该目录缓存。Compose 使用本地目录 bind mount，不依赖 Docker named volumes，方便打包、迁移和离线部署。

```bash
git clone git@github.com:muidea/magicNLP.git
cd magicNLP
cp .env.example .env
docker compose up --build -d
```

如果使用旧版 Docker Compose，把命令中的 `docker compose` 替换为 `docker-compose`。

检查服务：

```bash
curl http://127.0.0.1:8010/health
```

停止服务：

```bash
docker compose down
```

清理本地模型缓存时删除 `cache/huggingface/` 下的内容即可；目录本身由仓库保留。

## 快速开始：本地 Python

要求 Python 3.10+，推荐 Python 3.11。

```bash
git clone git@github.com:muidea/magicNLP.git
cd magicNLP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd nlpAPI
gunicorn -b 0.0.0.0:8080 --workers 1 --threads 4 --timeout 300 embed_server:app
```

本地开发也可以直接运行 Flask：

```bash
cd nlpAPI
python embed_server.py
```

## 配置

可通过环境变量配置运行参数：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | sentence-transformers 模型名或本地模型路径 |
| `LOCAL_FILES_ONLY` | `false` | 是否只使用本地 Hugging Face 缓存；离线部署设为 `true` |
| `EMBED_DEFAULT_INPUT_TYPE` | `passage` | 默认 embedding 输入类型，可选 `passage`、`query`、`raw` |
| `EMBED_QUERY_PREFIX` | `query: ` | `input_type=query` 时自动添加的前缀 |
| `EMBED_PASSAGE_PREFIX` | `passage: ` | `input_type=passage` 时自动添加的前缀 |
| `NLP_PORT` | `8010` | Docker Compose 暴露端口 |
| `GUNICORN_WORKERS` | `1` | Gunicorn worker 数量 |
| `GUNICORN_THREADS` | `4` | Gunicorn 每个 worker 的线程数 |
| `GUNICORN_TIMEOUT` | `300` | Gunicorn 请求超时时间，单位秒 |

中文优先、完全离线或已有缓存场景可在 `.env` 中切换模型：

```env
EMBED_MODEL=BAAI/bge-small-zh-v1.5
LOCAL_FILES_ONLY=false
```

离线部署时，先在联网环境启动一次服务让模型下载到 `cache/huggingface/`，再连同该目录一起拷贝到目标机器，启动时设置：

```env
LOCAL_FILES_ONLY=true
```

## 小型知识库/RAG 推荐用法

默认模型 `intfloat/multilingual-e5-small` 是 384 维向量，适合 DuckDB-VSS、Qdrant、Milvus-Lite 等轻量向量库。为了获得更好的中英文双语召回效果，建议区分文档和查询：

- 灌入知识库文档时传 `input_type=passage`，服务会自动添加 `passage: ` 前缀。
- 用户检索问题时传 `input_type=query`，服务会自动添加 `query: ` 前缀。
- 如果调用方已经自行添加前缀，服务会识别 `query: ` / `passage: `，不会重复添加。
- 需要完全原样向量化时传 `input_type=raw`。

## API 示例

### OpenAI 兼容 embeddings

```bash
curl http://127.0.0.1:8010/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "intfloat/multilingual-e5-small",
    "input_type": "passage",
    "input": ["DuckDB 是一个内存分析型数据库", "hello world"]
  }'
```

返回格式：

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.01, 0.02],
      "index": 0
    }
  ],
  "model": "intfloat/multilingual-e5-small",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

支持 `encoding_format=base64`：

```bash
curl http://127.0.0.1:8010/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "DuckDB 是一个内存分析型数据库",
    "input_type": "query",
    "encoding_format": "base64"
  }'
```

### Ollama 兼容 embed

```bash
curl http://127.0.0.1:8010/api/embed \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "intfloat/multilingual-e5-small",
    "input_type": "passage",
    "input": ["DuckDB 是一个内存分析型数据库", "hello world"]
  }'
```

### 历史单条 embedding

```bash
curl http://127.0.0.1:8010/api/v1/nlp_service/embedding/single \
  -H 'Content-Type: application/json' \
  -d '{"text":"DuckDB 是一个内存分析型数据库","input_type":"passage"}'
```

### 历史批量 embedding

```bash
curl http://127.0.0.1:8010/api/v1/nlp_service/embedding/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["DuckDB 是一个内存分析型数据库","hello world"],"input_type":"passage"}'
```

### 关键词提取

```bash
curl http://127.0.0.1:8010/api/v1/nlp_service/keywords/extract \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "DuckDB 是一个内存分析型数据库，适合实时分析和嵌入式分析场景",
    "top_k": 5,
    "business_context": "数据库 分析 OLAP",
    "threshold": 0.3
  }'
```

## OpenAI SDK 调用示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8010/v1", api_key="not-needed")

response = client.embeddings.create(
    model="intfloat/multilingual-e5-small",
    input=["DuckDB 是一个内存分析型数据库"],
    extra_body={"input_type": "query"}
)

print(len(response.data[0].embedding))
```

使用 SDK 示例需要额外安装：

```bash
pip install openai
```

## Go demo

启动服务后运行默认示例。默认会调用 OpenAI 兼容接口 `POST /v1/embeddings`：

```bash
go run ./nlpDemo
```

如果服务地址不是默认的 `http://127.0.0.1:8010`：

```bash
NLP_SERVER=http://127.0.0.1:8080 go run ./nlpDemo
```

也可以通过参数指定服务地址：

```bash
go run ./nlpDemo -server http://127.0.0.1:8010
```

支持的调用模式：

```bash
# 健康检查
go run ./nlpDemo -mode health

# OpenAI 兼容 embeddings
go run ./nlpDemo -mode openai -input-type passage -texts 'DuckDB 是一个内存分析型数据库|hello world'

# Ollama 兼容 embed
go run ./nlpDemo -mode ollama -input-type passage -texts 'DuckDB 是一个内存分析型数据库|hello world'

# 历史单条 embedding
go run ./nlpDemo -mode single -input-type query -text 'DuckDB 是一个内存分析型数据库'

# 历史批量 embedding
go run ./nlpDemo -mode batch -input-type passage -texts 'DuckDB 是一个内存分析型数据库|hello world'

# 关键词提取
go run ./nlpDemo -mode keywords \
  -text 'DuckDB 是一个内存分析型数据库，适合实时分析和嵌入式分析场景' \
  -top-k 5 \
  -context '数据库 分析 OLAP' \
  -threshold 0.3
```

查看全部参数：

```bash
go run ./nlpDemo -h
```

## 依赖

Python 运行依赖已在 `requirements.txt` 中声明：

- `flask`
- `gunicorn`
- `jieba`
- `numpy`
- `sentence-transformers`
- `torch`

Go demo 只依赖标准库。

`requirements.txt` 默认使用 PyTorch CPU wheel 源，适合普通服务器和本地开发。需要 CUDA/GPU 版本时，请按目标环境调整 `torch` 安装源后再构建镜像。

## 常见问题

### 首次启动很慢

首次启动会下载 embedding 模型，速度取决于网络和模型大小。下载完成后会使用本地缓存。

### 离线环境无法启动

如果设置了 `LOCAL_FILES_ONLY=true`，必须提前准备好模型缓存或把 `EMBED_MODEL` 指向本地模型目录。

### Docker Compose 端口冲突

修改 `.env`：

```env
NLP_PORT=8080
```

然后重新启动：

```bash
docker compose up -d
```
