package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
)

type EmbedRequest struct {
	Text string `json:"text"`
}

type EmbedResponse struct {
	Vector []float32 `json:"vector"`
}

func main() {
	text := "DuckDB 是一个内存分析型数据库"

	localServer := "http://127.0.0.1:8080"
	// remoteServer := "https://api.mulife.vip"
	// 构造请求
	reqBody, _ := json.Marshal(EmbedRequest{Text: text})
	resp, err := http.Post(fmt.Sprintf("%s/api/v1/nlp_service/embedding/single", localServer), "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		log.Fatal("请求失败:", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Fatalf("服务错误: %s", resp.Status)
	}

	// 解析响应
	var embedResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		log.Fatal("解析失败:", err)
	}

	fmt.Println("向量长度:", len(embedResp.Vector))
	fmt.Println("前5个维度:", embedResp.Vector[:5])
}
