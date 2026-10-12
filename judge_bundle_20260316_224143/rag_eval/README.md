# 以詞頻分區改善的 GraphRAG 評測實作

這個目錄提供一個以**詞頻導向分區（term-frequency-guided chunking）**改善檢索品質的 GraphRAG 研究原型與評測工具。核心想法是先分析文件中的高頻詞與句子邊界，再將內容切成較具主題一致性的區塊，降低固定長度切分造成的語意斷裂。

目前程式的主要流程包含：

1. 載入內建文件，或從 `../input/sample.txt` 載入自訂文件。
2. 以詞頻與句子邊界進行文件分區，必要時再退回固定長度重疊分塊。
3. 將分區後的文件加入知識庫並建立檢索索引。
4. 根據問題找出最相關的文件分區。
5. 將檢索結果放入 prompt，交給 OpenAI 相容的 Chat Completions API 產生答案。
6. 匯出查詢 trace、延遲與 CPU／記憶體／GPU 資源指標。
7. 以批次資料集比較詞頻分區 RAG、一般 RAG 與 no-RAG baseline 的表現。

> 這是 GraphRAG 的輕量研究原型。目前程式已實作「文件分區改善檢索」與可替換的檢索器，但尚未建立 entity/relation graph 或 graph traversal；若要完成完整 GraphRAG，後續仍需加入實體抽取、關係建圖與圖檢索階段。

## 目錄結構

```text
rag_eval/
├── rag.py                 # 詞頻分區、RAG 核心、檢索器、trace 與資源指標
├── evals.py               # 批次評測與結果輸出
├── pyproject.toml         # Python 專案與依賴宣告
├── uv.lock                # uv 鎖定的依賴版本
└── evals/
    ├── logs/              # 每次 RAG query 的 JSON trace
    └── experiments/       # 批次評測產生的 CSV
```

批次評測預設讀取：

```text
<judge_bundle>/evals/datasets/test_dataset_no_rag.csv
```

每一列應為：

```csv
question,grading note 1 - grading note 2
```

評測程式會以第一個逗號切分問題與評分備註，因此問題本身不應包含未經處理的第一個逗號。

## 需求

- Python 3.9 以上
- 一個 OpenAI 相容的 Chat Completions API
- `uv`（建議）或其他可安裝 Python 依賴的工具
- 預設模型名稱為 `mistral`

依賴由 [pyproject.toml](./pyproject.toml) 定義，主要包含：

- `openai`
- `psutil`
- `nvidia-ml-py`
- `ragas`

`psutil` 與 `nvidia-ml-py` 若無法載入，程式仍可執行，但對應的資源指標不會輸出。

## 安裝

在本目錄執行：

```powershell
uv sync
```

若不使用 `uv`，也可以使用一般 virtual environment，再安裝專案依賴：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## 設定 LLM

### 使用 Ollama（批次評測建議）

先啟動 Ollama，並確保本機已有 `mistral`：

```powershell
ollama pull mistral
```

`evals.py` 預設連線到：

```text
http://localhost:11434/v1
```

Ollama 不需要真正的 API key；程式會使用 `api_key="ollama"`。

### 使用 OpenAI 或其他 OpenAI 相容服務

批次評測可透過環境變數指定 endpoint 與模型：

```powershell
$env:RAG_EVAL_BASE_URL = "https://api.openai.com/v1"
$env:RAG_EVAL_MODEL = "gpt-4o-mini"
$env:OPENAI_API_KEY = "<your-api-key>"
```

> `evals.py` 的 client 目前固定使用 `api_key="ollama"`。若服務端需要驗證，請先在程式中將 client 的 `api_key` 改為讀取 `OPENAI_API_KEY`，或使用不要求真實 key 的 OpenAI 相容服務。

## 快速開始：單次 RAG 查詢

`rag.py` 直接執行時使用 `OPENAI_API_KEY`，並連線到 OpenAI 預設 endpoint：

```powershell
$env:OPENAI_API_KEY = "<your-openai-api-key>"
python .\rag.py
```

程式會：

- 使用內建文件；
- 使用 `SimpleKeywordRetriever`；
- 查詢 `What is Ragas`；
- 將 trace 寫入 `logs\`；
- 在終端機印出答案與 log 路徑。

如果要在 Python 程式中使用 RAG client：

```python
from openai import OpenAI

from rag import default_rag_client

llm = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
rag = default_rag_client(llm_client=llm, logdir="logs")

result = rag.query("What is retrieval-augmented generation?", top_k=3)

print(result["answer"])
print(result["retrieved_docs"])
print(result["metrics"])
print(result["logs"])
```

`query()` 回傳的主要欄位：

| 欄位 | 說明 |
| --- | --- |
| `answer` | LLM 產生的答案 |
| `run_id` | 本次查詢的識別碼 |
| `retrieved_docs` | 實際取回的文件、文件編號與相似度 |
| `metrics` | 查詢與 LLM 延遲、資源起訖快照及差值 |
| `logs` | JSON trace 檔案路徑 |

## 執行批次評測

在 `rag_eval` 目錄執行：

```powershell
python .\evals.py
```

預設設定：

- `RAG_EVAL_MODE=rag`
- `RAG_EVAL_MODEL=mistral`
- `RAG_EVAL_BASE_URL=http://localhost:11434/v1`
- 啟用 noise sensitivity
- 每題加入 2 個隨機無關文件
- 使用向量檢索器

完成後會在以下位置產生檔案：

```text
evals/logs/*.json
evals/experiments/eval_results_<mode>_<model>_<timestamp>.csv
```

### RAG 與 no-RAG 比較

RAG：

```powershell
$env:RAG_EVAL_MODE = "rag"
python .\evals.py
```

no-RAG baseline（直接讓 LLM 回答，不進行檢索）：

```powershell
$env:RAG_EVAL_MODE = "no_rag"
python .\evals.py
```

### 切換檢索器

預設使用 TF-IDF 向量檢索：

```powershell
$env:RAG_RETRIEVER = "vector"
python .\evals.py
```

切換成簡單關鍵字匹配：

```powershell
$env:RAG_RETRIEVER = "keyword"
python .\evals.py
```

### 使用自訂文件

將自訂內容放在：

```text
../input/sample.txt
```

然後啟用：

```powershell
$env:RAG_EVAL_USE_SAMPLE_TXT = "1"
python .\evals.py
```

預設會先以句子邊界與 term frequency 將內容分區，再建立索引。這是本專案的主要改善方法：分區時會統計文件詞頻，利用區塊內詞頻變化尋找較自然的主題邊界；若分區結果不足，則退回固定長度、帶 200 字元 overlap 的分塊。

若要停用詞頻分區、將整個檔案視為單一文件：

```powershell
$env:RAG_EVAL_DISABLE_CHUNKING = "1"
python .\evals.py
```

## 環境變數

| 變數 | 預設值 | 用途 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 無 | `rag.py` 直接執行時的 API key |
| `RAG_EVAL_MODE` | `rag` | 評測模式：`rag` 或 `no_rag` |
| `RAG_EVAL_MODEL` | `mistral` | 評測使用的模型名稱 |
| `RAG_EVAL_BASE_URL` | `http://localhost:11434/v1` | OpenAI 相容 API endpoint |
| `RAG_RETRIEVER` | `vector` | `vector` 或 `keyword` |
| `RAG_EVAL_USE_SAMPLE_TXT` | `0` | 是否載入 `../input/sample.txt` |
| `RAG_EVAL_DISABLE_CHUNKING` | `0` | 是否停用自訂文件分塊 |
| `RAG_EVAL_NOISE_ENABLE` | `1` | 是否執行 noise sensitivity |
| `RAG_EVAL_NOISE_DOCS` | `2` | 每題加入的無關文件數量 |

PowerShell 範例：

```powershell
$env:RAG_EVAL_MODE = "rag"
$env:RAG_RETRIEVER = "keyword"
$env:RAG_EVAL_NOISE_ENABLE = "0"
$env:RAG_EVAL_MODEL = "mistral"
python .\evals.py
```

## 評測指標與輸出欄位

`evals.py` 會對每個問題產生一列 CSV，主要指標如下：

- `score`：由 LLM 判斷答案是否涵蓋 grading notes，值為 `pass` 或 `fail`。
- `relevance_score`：由 LLM 判斷取回文件是否與問題相關；no-RAG 時為 `n/a`。
- `context_recall_score`：grading note 項目出現在取回文件中的比例，範圍為 `0..1`。
- `context_recall_missing`：未在取回文件找到的 grading note 項目。
- `noise_sensitivity_score`：加入無關文件後是否仍維持正確；若答案由 pass 變 fail，記為 `no`。
- `noise_correctness_score`：加入 noise 後的 correctness 結果。
- `query_latency_ms`、`llm_latency_ms`：查詢與 LLM 呼叫耗時。
- `proc_rss_mb_delta`、`proc_cpu_time_s_delta`：程序記憶體與 CPU time 差值。
- `gpu_util_percent_delta`：若 NVML 可用，記錄 GPU 使用率差值。

## Trace 日誌

每次 `query()` 會在 `logdir` 產生一個 JSON 檔，內容包含：

- 查詢與 `run_id`
- 使用的文件數
- `query_start`、`retrieval`、`llm_call`、`llm_response`、`query_complete` 等事件
- 每次 LLM 呼叫的延遲與 token usage（API 有回傳時）
- CPU、記憶體及 GPU 資源快照（可取得時）
- 錯誤事件與錯誤訊息

這些日誌適合用來分析檢索結果、prompt context、延遲與資源使用；`evals/logs/` 與 `evals/experiments/` 下的結果檔通常應視為產物，不建議手動修改。

## 程式元件

### 詞頻導向文件分區

`_chunk_text_by_term_frequency()` 會先將文件拆成句子／換行單位，再計算整份文件的詞頻。當目前區塊達到最小長度，且下一段與目前區塊的詞頻關聯顯著降低時，就結束目前區塊。每個區塊上限為 1500 字元、下限為 600 字元；若無法產生至少兩個有效區塊，則退回 `_chunk_text()` 的固定長度重疊分塊。

這個方法的預期效果是：

- 讓同一主題的句子盡量留在同一個檢索單位；
- 降低關鍵資訊剛好落在固定長度邊界而被切開的機率；
- 在不引入 embedding model 的前提下，改善 lexical retrieval 的 context quality；
- 讓後續 entity/relation extraction 更容易以主題區塊作為輸入。

### `SimpleKeywordRetriever`

以 query 與文件的 token 交集數量排序。支援英文／數字 token、中文單字與中文 bigram。優點是簡單、快速且不需要額外模型；缺點是無法處理同義詞或語意相近但字面不同的內容。

### `SimpleVectorRetriever`

建立 TF-IDF 文件向量，使用 cosine similarity 排序。它仍是本地 lexical retrieval，不是 embedding retrieval；不會呼叫 embedding API。詞頻分區會在此檢索階段之前改變文件單位，因此可以直接比較「原始文件」與「詞頻分區文件」的效果。

### `ExampleRAG`

提供以下主要操作：

- `add_documents(documents)`：附加文件並重新 fit。
- `add_documents_parallel(documents)`：平行正規化後附加文件。
- `set_documents(documents)`：取代現有文件並重新 fit。
- `retrieve_documents(query, top_k=3)`：只執行檢索。
- `generate_response(query, top_k=3)`：檢索後呼叫 LLM。
- `query(question, top_k=3, run_id=None)`：執行完整流程並輸出 trace。

## 注意事項

- `top_k` 只是候選文件上限；相似度為 0 的文件會被排除，因此實際取回數量可能少於 `top_k`。
- 預設模型名稱在 `rag.py` 為固定的 `mistral`；若使用其他模型，請確認 API server 支援該模型名稱。
- `rag.py` 直接執行與 `evals.py` 的 endpoint 設定不同：前者使用 OpenAI 預設 endpoint，後者預設使用 Ollama。
- `query()` 會將例外轉成結果中的錯誤文字並寫入 trace；批次評測仍會繼續處理後續資料列。
- 評測中的 correctness 與 relevance 是由 LLM judge 判斷，結果會受 judge 模型與 prompt 影響，不能視為完全客觀的基準。
- GPU 指標只會嘗試讀取第 0 張 GPU；沒有 NVIDIA GPU、NVML 不可用或初始化失敗時，相關欄位會缺省。

## 建議實驗設計

若要驗證「詞頻分區改善 GraphRAG」的效果，建議至少執行下列組合，並比較同一份資料集的 `score`、`relevance_score`、`context_recall_score` 與延遲：

| 實驗組別 | `RAG_EVAL_MODE` | `RAG_EVAL_DISABLE_CHUNKING` | 說明 |
| --- | --- | --- | --- |
| No-RAG baseline | `no_rag` | 不適用 | 不使用檢索，直接回答 |
| 一般 RAG | `rag` | `1` | 不做詞頻分區，作為切分基準 |
| 詞頻分區 RAG | `rag` | `0` | 本專案的主要方法 |

例如：

```powershell
# 一般 RAG baseline
$env:RAG_EVAL_MODE = "rag"
$env:RAG_EVAL_DISABLE_CHUNKING = "1"
python .\evals.py

# 詞頻分區 RAG
$env:RAG_EVAL_MODE = "rag"
$env:RAG_EVAL_DISABLE_CHUNKING = "0"
python .\evals.py
```

兩次執行會產生不同檔名的 CSV；檔名中的 `rag_nochunk` 與 `rag_chunked` 可用來區分實驗組別。
