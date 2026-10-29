# 2B

**小而能打的多模态模型在 8GB 显存（RTX 3070）上的真实可行性研究**

> 两件事：
> ① **估算**——参数量 1B–2B 的模型，在 8GB 消费卡上跑不跑得动，判据是「峰值显存」而不是「权重体积」；
> ② **实测**——五个文档解析引擎在同一台 3070、同一批 PDF 上的真实数字。

---

## 一句话结论

### 估算侧（17 个模型，BF16 峰值 + 厂商硬性要求）

**A 开箱即跑 4 个**（HunyuanOCR、GLM-OCR、Qwen3-ASR-0.6B、MiniCPM-OCR Jev）
**B 可行需限制 9 个**（要控分辨率、音频长度或上下文）
**C 边缘 3 个**（FireRed-OCR、Granite Speech、Qwen3-ASR-1.7B，必须量化）
**D 不可行 2 个**（OXR 本地需 24GB；VoxCPM2 实测峰值 7.3–7.9GB 超 3070 可用显存）

### 实测侧（RTX 3070，两篇期刊 PDF，五引擎）

**没有赢家。** 每个引擎的强项和硬伤完全不重叠：

| 要什么 | 选谁 | 代价 |
|---|---|---|
| 全量入库、数据不出内网 | MinerU 8.4 s/页 | 扫描件术语错字「压人模量」 |
| RAG 按章节切块 | MiniCPM-OCR 15.3 s/页 | 必须关图表识别，否则编数据 |
| 保留原图 | TeleOCR 65 s/页 | 慢 8 倍，带水印 |
| 只要文字、最准 | OXR 10.3 s/页 | 图链 16 分钟过期，本地跑不了 |

详见 [docs/03-五引擎实测.md](docs/03-五引擎实测.md)。

---

## 一、先分清三个数

| 维度 | 含义 | 怎么算 | 典型误差来源 |
|---|---|---|---|
| **参数量** | 模型有多少参数（如 1.3B） | 模型卡 / safetensors 元数据 | 官方标的常只算语言主干，**不含视觉/音频编码器** |
| **权重体积** | 落在磁盘上的文件大小 | 参数量 × 每参字节数 | BF16 = 2B/参，Q8 ≈ 1B/参，Q4 ≈ 0.5B/参 |
| **峰值显存** | 推理那一刻 GPU 上真正被占用的量 | 权重 + KV Cache + 激活 + 框架开销 | **决定能不能跑的是这个** |

```
峰值显存 = 权重
         + KV Cache      (2 × 层数 × KV头数 × head_dim × 字节 × token 数 × 并发数)
         + 激活/中间张量  (随分辨率、音频长度、batch 线性增长)
         + 框架开销      (CUDA context ≈ 0.3-0.5GB；vLLM 默认预分配 90% 显存)
```

**3070 8GB 的可支配显存约 7.2GB**（Windows 桌面合成常驻 0.5–0.8GB），不是 8GB。

---

## 二、模型矩阵（估算）

分级口径（绝对阈值）：**A** ≤4.0GB · **B** 4.0–6.2GB · **C** 6.2–7.2GB · **D** >7.2GB 或厂商最低要求超 8GB

| # | 模型 | 任务 | 参数 | BF16 权重 | Q4 权重 | 峰值显存 | 3070 | 实测 |
|---|---|---|---|---|---|---|---|---|
| 1 | HunyuanOCR | OCR | 1.0B | 2.0GB | 0.6GB | 3.0–3.8GB | **A** | — |
| 2 | GLM-OCR | OCR | 0.9B | 1.8GB | 0.55GB | 4.0GB ⚠️官方 | **A** | — |
| 3 | Qwen3-ASR-0.6B | ASR | 0.94B | 1.8GB | 0.5GB | 2.1–3.9GB | **A** | — |
| 4 | MiniCPM-OCR Jev | OCR 质检 | 2.0B | 4.0GB | 1.94GB | 2.5–3.5GB | **A** | 🟡 进行中 |
| 5 | LFM3-1.7B | 文本 | 1.73B | 3.5GB | 1.0GB | 1.0–4.3GB | **B** | — |
| 6 | MiniCPM-OCR 1.0 | OCR | 1.3B | 2.42GB | 1.43GB | 3.0–4.4GB | **B** | ✅ 15.3 s/页 |
| 7 | TeleOCR | OCR | 1.2B | 2.83GB | 0.8GB | 4.0–6.0GB | **B** | ✅ 65 s/页 |
| 8 | LFM3-Audio-2B | ASR+TTS | 2.0B | 4.0GB | 1.1GB | 1.0–5.3GB | **B** | — |
| 9 | Qwen3.5-2B | 文本+视觉 | 2.27B | 4.0GB | 1.28GB | 2.2–5.7GB | **B** | — |
| 10 | GLM-ASR-Nano-2512 | ASR | 1.5B | 3.5GB | 1.0GB | 4.5–5.9GB | **B** | — |
| 11 | Cohere Transcribe 03-2026 | ASR | 2.1B | 4.2GB | 1.21GB | 4.6–6.0GB | **B** | — |
| 12 | Qwen3-VL-Embedding-2B | Embedding | 2.1B | 4.4GB | 1.1GB | 4.5–6.2GB | **B** | — |
| 13 | Qwen3-ASR-1.7B | ASR | 2.3B | 3.9GB | 1.2GB | 4.0–6.4GB | **C** | — |
| 14 | Granite Speech 4.1 2B | ASR/AST | 2.3B | 4.3GB | 2.14GB | 3.0–6.4GB | **C** | — |
| 15 | FireRed-OCR | OCR | 2.13B | 4.3GB | 1.3GB | 6.0–7.2GB | **C** | — |
| 16 | VoxCPM2 | TTS | 2.3B | 5.0GB | 1.1GB | 7.3–8.0GB | **D** | — |
| 17 | OXR-1.0 | OCR | 1.3B | 3.2GB | 1.0GB | 24GB+ ⚠️官方 | **D** | ✅ 云端 10.3 s/页 |

> 机器可读版：[`data/models.csv`](data/models.csv)，含 `evidence`（证据分级）、`hard_min_vram_gb`（厂商硬性要求）、
> `verified`（local/cloud/not-tested）、`bench_s_per_page`（实测速度）。
> 上表数字来自 `python scripts/vram_estimate.py --csv data/models.csv --ctx 8192`。

⚠️ = 厂商明确给出的最低显存，优先级高于公式。✅ = 本机或云端实测过。

---

## 三、原表的 5 处硬伤（已修正）

### 1. 「极充裕（GGUF Q8 约 1.7GB）」低估了实际占用

MiniCPM-OCR 的 1.7GB 是 **LLM Q8_0（0.812GB）+ 视觉投影 mmproj（0.901GB）** 两块相加——投影层占一半。
llama.cpp 只传 `--model` 不传 `--mmproj` 会直接报错。**它的 A 档不成立，是 B 档。**

```bash
llama-server --model MiniCPM-OCR-1_0-Q8_0.gguf --mmproj mmproj-model-f16-4x.gguf \
  --ctx-size 8192 --n-gpu-layers 999
```

### 2. FireRed-OCR 不是 1.3B，是 2.13B

基于 Qwen3-VL-2B-Instruct。BF16 权重 4.3GB + 视觉激活 → 6.0–7.2GB，**C 档**。

### 3. GLM-OCR 官方自己标了 4GB 运行显存

不是"极充裕"。CogViT 视觉编码器吃高分辨率图像，官方给的 4GB 才是可用数字。

### 4. OXR 本地跑不了，但云端 API 完全可用（实测修正）

官方 README 要求 **24GB 起、推荐 48GB**，**只支持 Linux**，无 llama.cpp/Ollama 路径。
权重只有 3.2GB —— "权重小 ≠ 能跑"最典型的反例。
**但免费 API 是现成能用的**（`https://free-oxr.somark.cn/api/v1`，无需注册），实测 10.3 s/页、还原率 98.3%、零错字。

### 5. VoxCPM2 实测峰值 7.3–7.9GB

官方"建议 8GB"是门槛不是舒适区，3070 可用只有 7.2GB。**BF16 档会 OOM**，必须走 q8（2.3GB）或 4-bit（1.1GB）。

### 实测带来的一处推翻

**TeleOCR 不是"必须 vLLM、Windows 跑不了"。** 代码里有完整 `transformers` 分支（README 未更新），
架构是 **Qwen2.5-VL 视觉编码器 + Qwen3 文本解码器**（QK-Norm、无 bias），
只需 transformers ≥ 4.57，2.83GB 权重在 3070 上**显存 < 8GB 跑通**，65 s/页。
详见 [docs/03](docs/03-五引擎实测.md)。

---

## 四、部署栈：3070 上最容易翻车的地方

| 栈 | 显存行为 | 适合 8GB 卡？ |
|---|---|---|
| **llama.cpp / Ollama** | 按需分配，`n-gpu-layers` 可只卸载部分层 | ✅ 首选 |
| **Transformers** | 权重 + 激活按需，无预分配 | ✅ 可以，注意 `max_memory` 与 `device_map` |
| **vLLM / SGLang** | `gpu_memory_utilization` 默认 **0.9**，启动就把 90% 显存划进 KV 池 | ⚠️ 必须显式改 |

```bash
vllm serve <model> --gpu-memory-utilization 0.75 --max-model-len 8192 \
  --max-num-seqs 4 --enforce-eager
```

---

## 五、三个最贵的坑（实测得出，可直接复用）

### 1. MiniCPM-OCR 默认配置会编造表格数据

`use_chart_recognition: true` 把折线图转成"数据表"并**编造数值**。两份无表 PDF 输出了 8 个假表。

| 配置 | 假表格 | 字符数 |
|---|---|---|
| 默认（true） | 6 | 17,892 |
| `false` | **0** | 15,970 |

少掉的 1,922 字符全是编造内容。**入库前必须关掉**：

```bash
--set layout.use_chart_recognition=false
```

### 2. TeleOCR 批次推理必须改 left padding

默认 right-padding 会污染短序列，首页英文标题块吐出 2028 字符的 `of 100mls of 100mls…` 幻觉。
同一 bbox 单独跑（batch=1）输出完全正确。改 `padding_side="left"` 后 **157 个块零异常**。
另外 `max_new_tokens=None` 会退化成 `max_length=128000`，必须手动设。

### 3. OXR 的图链只有约 16 分钟寿命

`/api/v1/files/...png` 会 307 跳转到腾讯 COS 临时签名 URL（`q-sign-time` 到 `q-key-time` 差 960 秒）。
批量跑不即时下载，Markdown 里全是死链。

补充：**Windows 装 Ollama 用 `/S` 不是 `/VERYSILENT`**（它是 NSIS 不是 Inno Setup），
配 `MSYS_NO_PATHCONV=1` 防止 Git Bash 转换路径。更多见 [docs/03 第七节](docs/03-五引擎实测.md)。

---

## 六、研究路线

### Phase 0 — 基建（1 天）
- [x] 跑 `scripts/vram_estimate.py` 生成基线表
- [x] 统一测量口径（峰值显存 + 单页耗时）
- [ ] 固定环境快照：驱动 / CUDA / torch / transformers 版本

### Phase 1 — 纯文本 + Embedding 打底（2–3 天）
Qwen3.5-2B → LFM3-1.7B → Qwen3-VL-Embedding-2B
- 目标：拿到干净的「参数量 → 峰值显存」基线，量化估算公式误差
- 变量：量化档位 × 上下文 × batch

### Phase 2 — 文档解析（进行中，已完成 5 个）
- [x] OXR（云端 API）
- [x] MinerU 1.3.12（本地，最快）
- [x] Unlimited-OCR（本地，最慢）
- [x] TeleOCR（本地，唯一图落盘）
- [x] MiniCPM-OCR（本地，层级 + 公式最好）
- [ ] **表格维度 —— 最大空白，5 个引擎一个都没测到**

### Phase 3 — 语音（2 周）
Qwen3-ASR-0.6B → Granite Speech → Cohere Transcribe → GLM-ASR → LFM3-Audio → VoxCPM2

### Phase 4 — 质检与产出
- [ ] MiniCPM-OCR Jev 接进流水线，做「哪些页需人工复核」分诊
- [ ] `bench/<model>.md` 逐模型记录
- [ ] 一张「输入规模 vs 显存」总图

---

## 七、复现

```bash
git clone https://github.com/943fansi/2B.git
cd 2B
python scripts/vram_estimate.py --csv data/models.csv --gpu-mem 8 --ctx 8192
python scripts/vram_estimate.py --model qwen3.5-2b --ctx 32768
python scripts/vram_estimate.py --csv data/models.csv --precision q4
```

脚本只做**静态估算**（不加载权重），用于在下载几十 GB 之前先判断可行性。

---

## 八、数据来源与可信度

| 标记 | 含义 | 可信度 |
|---|---|---|
| **本机实测** | 同一台 3070、同一批 PDF 跑出的数字 | 最高 |
| 官方文件体积 / 文档 / README | 从 HF 仓库文件大小或厂商文档读到 | 高 |
| 第三方实测 | 独立评测站点的实测峰值 | 中 |
| 自行推算 | 按公式从参数量推导 | 低，需实测校准 |

**所有 `peak_vram_est_gb` 都是估算，不是保证。** `bench_s_per_page` 才是实测值。

---

## License

文档部分 CC BY 4.0。各模型权重遵循其原始许可（见 `data/models.csv` 的 `license` 列）。
