package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

const defaultText = "DuckDB 是一个内存分析型数据库"

type config struct {
	server          string
	mode            string
	model           string
	inputType       string
	text            string
	texts           string
	topK            int
	businessContext string
	threshold       float64
	timeout         time.Duration
}

type legacySingleRequest struct {
	Text      string `json:"text"`
	InputType string `json:"input_type,omitempty"`
}

type legacySingleResponse struct {
	Vector []float64 `json:"vector"`
}

type legacyBatchRequest struct {
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type,omitempty"`
}

type legacyBatchResponse struct {
	Vectors [][]float64 `json:"vectors"`
}

type openAIEmbeddingRequest struct {
	Model     string   `json:"model,omitempty"`
	Input     []string `json:"input"`
	InputType string   `json:"input_type,omitempty"`
}

type openAIEmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type ollamaEmbedRequest struct {
	Model     string   `json:"model,omitempty"`
	Input     []string `json:"input"`
	InputType string   `json:"input_type,omitempty"`
}

type ollamaEmbedResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float64 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration"`
	LoadDuration    int64       `json:"load_duration"`
	PromptEvalCount int         `json:"prompt_eval_count"`
}

type keywordRequest struct {
	Text            string  `json:"text"`
	TopK            int     `json:"top_k,omitempty"`
	BusinessContext string  `json:"business_context,omitempty"`
	Threshold       float64 `json:"threshold,omitempty"`
}

type keywordResponse struct {
	Keywords []string `json:"keywords"`
}

type healthResponse struct {
	Status           string `json:"status"`
	Model            string `json:"model"`
	DefaultInputType string `json:"default_input_type"`
}

func main() {
	cfg := parseFlags()

	client := &http.Client{Timeout: cfg.timeout}
	switch cfg.mode {
	case "health":
		runHealth(client, cfg)
	case "single":
		runLegacySingle(client, cfg)
	case "batch":
		runLegacyBatch(client, cfg)
	case "openai":
		runOpenAIEmbedding(client, cfg)
	case "ollama":
		runOllamaEmbed(client, cfg)
	case "keywords":
		runKeywords(client, cfg)
	default:
		log.Fatalf("不支持的 mode: %s，可选值: health, single, batch, openai, ollama, keywords", cfg.mode)
	}
}

func parseFlags() config {
	server := os.Getenv("NLP_SERVER")
	if server == "" {
		server = "http://127.0.0.1:8010"
	}

	cfg := config{}
	flag.StringVar(&cfg.server, "server", server, "magicNLP 服务地址，也可用 NLP_SERVER 环境变量")
	flag.StringVar(&cfg.mode, "mode", "openai", "调用模式: health, single, batch, openai, ollama, keywords")
	flag.StringVar(&cfg.model, "model", "intfloat/multilingual-e5-small", "请求中携带的模型名")
	flag.StringVar(&cfg.inputType, "input-type", "passage", "embedding 输入类型: passage, query, raw")
	flag.StringVar(&cfg.text, "text", defaultText, "单条输入文本")
	flag.StringVar(&cfg.texts, "texts", "", "多条输入文本，使用 | 分隔；为空时使用 -text 和 hello world")
	flag.IntVar(&cfg.topK, "top-k", 5, "关键词提取数量")
	flag.StringVar(&cfg.businessContext, "context", "数据库 分析 OLAP", "关键词提取的业务语境")
	flag.Float64Var(&cfg.threshold, "threshold", 0.3, "关键词提取的业务语境相似度阈值")
	flag.DurationVar(&cfg.timeout, "timeout", 60*time.Second, "HTTP 请求超时时间")
	flag.Parse()

	cfg.server = strings.TrimRight(cfg.server, "/")
	cfg.mode = strings.ToLower(strings.TrimSpace(cfg.mode))
	return cfg
}

func runHealth(client *http.Client, cfg config) {
	var response healthResponse
	getJSON(client, cfg.server+"/health", &response)
	fmt.Printf("服务状态: %s\n模型: %s\n默认输入类型: %s\n", response.Status, response.Model, response.DefaultInputType)
}

func runLegacySingle(client *http.Client, cfg config) {
	var response legacySingleResponse
	postJSON(client, cfg.server+"/api/v1/nlp_service/embedding/single", legacySingleRequest{Text: cfg.text, InputType: cfg.inputType}, &response)
	printVectorSummary("历史单条 embedding", response.Vector)
}

func runLegacyBatch(client *http.Client, cfg config) {
	var response legacyBatchResponse
	postJSON(client, cfg.server+"/api/v1/nlp_service/embedding/batch", legacyBatchRequest{Texts: inputTexts(cfg), InputType: cfg.inputType}, &response)

	fmt.Printf("历史批量 embedding: %d 条\n", len(response.Vectors))
	for index, vector := range response.Vectors {
		printVectorSummary(fmt.Sprintf("  #%d", index), vector)
	}
}

func runOpenAIEmbedding(client *http.Client, cfg config) {
	var response openAIEmbeddingResponse
	request := openAIEmbeddingRequest{
		Model:     cfg.model,
		Input:     inputTexts(cfg),
		InputType: cfg.inputType,
	}
	postJSON(client, cfg.server+"/v1/embeddings", request, &response)

	fmt.Printf("OpenAI embeddings: model=%s object=%s prompt_tokens=%d\n", response.Model, response.Object, response.Usage.PromptTokens)
	for _, item := range response.Data {
		printVectorSummary(fmt.Sprintf("  index=%d", item.Index), item.Embedding)
	}
}

func runOllamaEmbed(client *http.Client, cfg config) {
	var response ollamaEmbedResponse
	request := ollamaEmbedRequest{
		Model:     cfg.model,
		Input:     inputTexts(cfg),
		InputType: cfg.inputType,
	}
	postJSON(client, cfg.server+"/api/embed", request, &response)

	fmt.Printf("Ollama embed: model=%s prompt_eval_count=%d total_duration=%dns\n", response.Model, response.PromptEvalCount, response.TotalDuration)
	for index, vector := range response.Embeddings {
		printVectorSummary(fmt.Sprintf("  #%d", index), vector)
	}
}

func runKeywords(client *http.Client, cfg config) {
	var response keywordResponse
	request := keywordRequest{
		Text:            cfg.text,
		TopK:            cfg.topK,
		BusinessContext: cfg.businessContext,
		Threshold:       cfg.threshold,
	}
	postJSON(client, cfg.server+"/api/v1/nlp_service/keywords/extract", request, &response)
	fmt.Printf("关键词: %s\n", strings.Join(response.Keywords, ", "))
}

func inputTexts(cfg config) []string {
	if strings.TrimSpace(cfg.texts) == "" {
		return []string{cfg.text, "hello world"}
	}

	parts := strings.Split(cfg.texts, "|")
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		text := strings.TrimSpace(part)
		if text != "" {
			texts = append(texts, text)
		}
	}
	if len(texts) == 0 {
		log.Fatal("-texts 没有有效文本")
	}
	return texts
}

func getJSON(client *http.Client, url string, response any) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		log.Fatalf("构造请求失败: %v", err)
	}
	doJSON(client, req, response)
}

func postJSON(client *http.Client, url string, payload any, response any) {
	body, err := json.Marshal(payload)
	if err != nil {
		log.Fatalf("编码请求失败: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		log.Fatalf("构造请求失败: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	doJSON(client, req, response)
}

func doJSON(client *http.Client, req *http.Request, response any) {
	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("请求失败: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("读取响应失败: %v", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		log.Fatalf("服务返回错误: %s\n%s", resp.Status, string(body))
	}
	if err := json.Unmarshal(body, response); err != nil {
		log.Fatalf("解析响应失败: %v\n%s", err, string(body))
	}
}

func printVectorSummary(title string, vector []float64) {
	previewSize := 5
	if len(vector) < previewSize {
		previewSize = len(vector)
	}
	fmt.Printf("%s: 维度=%d 前%d维=%v\n", title, len(vector), previewSize, vector[:previewSize])
}
