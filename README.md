# 🌍 `open-translate`

[![Deploy on RunPod](https://img.shields.io/badge/Deploy%20on-RunPod-5d29f0?style=for-the-badge&logo=runpod)](https://console.runpod.io/deploy?template=s2nhyzkvdh&ref=464fs7zk)
[![Docker Hub](https://img.shields.io/docker/pulls/ivanvmoreno/open-translate?style=for-the-badge&logo=docker)](https://hub.docker.com/repository/docker/ivanvmoreno/open-translate/general)

> **A high-performance, self-hostable translation API compatible with Google Cloud Translate.**
> Built on Meta's **NLLB-200** and optimized with **DeepSpeed** for efficient GPU inference.

---

## ✨ Why?

This project provides a robust, private, and cost-effective alternative to commercial translation APIs.

* **💰 Cost Efficiency:** Run on your own GPU infrastructure. Ideal for high-volume translation tasks.
* **🔒 Data Privacy:** No external API calls mean your content never leaves your control.
* **🔄 Drop-in Compatibility:** Implements the standard `POST /language/translate/v2` API surface. Switch existing applications simply by changing the base URL.
* **🌍 Advanced Models:** Leverages Meta's [NLLB-200 (No Language Left Behind)](https://arxiv.org/abs/2207.04672), supporting 200+ languages.
* **🚀 High Performance:** Optimized for throughput with DeepSpeed and Tensor Parallelism, capable of handling heavy concurrent loads.

---

## ⚡ Drop-in Replacement

Designed to work with existing Google Cloud Translate client libraries and integrations.

**Before:**
`https://translation.googleapis.com/language/translate/v2`

**After:**
`http://localhost:8000/language/translate/v2`

---

## 🚀 Quick Start

### 🐳 Run with Docker

This command launches the API on port `8000` using the `600M` distilled model.

```bash
docker pull ivanvmoreno/open-translate:latest
docker run --gpus all -p 8000:8000 \
  -e NLLB_MODEL_SIZE=600M \
  -e DTYPE=fp16 \
  ivanvmoreno/open-translate:latest
```

> **Note:** The first run downloads the model weights, which may take some time depending on your internet speed.

---

## 🛠️ API Reference

Compatible with **Google Cloud Translation API v2**.

### Translate Text

**POST** `/language/translate/v2`

**Single Translation:**

```bash
curl -X POST "http://localhost:8000/language/translate/v2" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "Hello world!",
    "target": "es"
  }'
```

**Batch Translation:**
Send arrays of strings to maximize GPU throughput.

```bash
curl -X POST "http://localhost:8000/language/translate/v2" \
  -H "Content-Type: application/json" \
  -d '{
    "q": ["Hello world!", "Self hosting rulez"],
    "target": "fr",
    "source": "en",
    "max_new_tokens": 128
  }'
```

### Language Detection

**POST** `/language/translate/v2/detect`

```bash
curl -X POST "http://localhost:8000/language/translate/v2/detect" \
  -H "Content-Type: application/json" \
  -d '{"q": "Hola mundo"}'
```

### List Supported Languages

**GET** `/language/translate/v2/languages`

```bash
curl "http://localhost:8000/language/translate/v2/languages"
```

---

## ⚙️ Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NLLB_MODEL_SIZE` | `1.3B-distilled` | Model size: `600M`, `600M-distilled`, `1.3B`, `1.3B-distilled`, or `3.3B` |
| `NLLB_MODEL_ID` | *(None)* | HF model override |
| `TP_SIZE` | `auto` | Tensor Parallel size |
| `DTYPE` | `fp16` | `fp16`, `bf16`, or `fp32` |
| `MAX_BATCH_SIZE` | `32` | Max sentences processed in parallel |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |

---

## 🌐 Language Codes

We support standard **ISO 639-1** (e.g., `es`, `en`) and **BCP-47** (e.g., `zh-TW`, `pt-BR`) codes, automatically mapping them to NLLB's internal representation.

For a full list of over 200 supported languages and their codes, see **[LANGUAGES.md](./LANGUAGES.md)**.

---

## 💾 VRAM Requirement Guide

| Model Size      | FP16 / BF16 | FP32    |
|-----------------|-------------|---------|
| `600M` / `600M-distilled` | ~3 GB      | ~5 GB  |
| `1.3B` / `1.3B-distilled` | ~5 GB      | ~9 GB |
| `3.3B`           | ~9 GB     | ~15 GB |
