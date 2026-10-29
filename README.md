# 2B

**小而能打的多模态模型在 8GB 显存（RTX 3070）上的真实可行性研究**

> 本仓库只关心一件事：**参数量 1B–2B 的模型，在 8GB 消费级显卡上，到底跑不跑得动。**
> 判断依据是「峰值显存」而不是「权重文件体积」——这两个数不是一回事，差得很远。

---

## 一句话结论

16 个模型，按 **BF16 峰值显存 + 厂商硬性最低要求** 分级：

**A 开箱即跑 3 个**（HunyuanOCR、GLM-OCR、Qwen3-ASR-0.6B）
**B 可行需限制 8 个**（OCR/ASR 主力，要控分辨率、音频长度或上下文）
**C 边缘 3 个**（FireRed-OCR、Granite Speech、Qwen3-ASR-1.7B，必须量化）
**D 不可行 2 个**（OXR 官方要求 24GB；VoxCPM2 实测峰值 7.3–7.9GB 超 3070 可用显存）

---

## 一、先分清三个数

| 维度 | 含义 | 怎么算 | 典型误差来源 |
|---|---|---|---|
| **参数量** | 模型有多少参数（如 1.3B） | 模型卡 / safetensors 元数据 | 官方标的常只算语言主干，**不含视觉/音频编码器** |
| **权重体积** | 落在磁盘上的文件大小 | 参数量 × 每参字节数 | BF16 = 2B/参，Q8 ≈ 1B/参，Q4 ≈ 0.5B/参 |
| **峰值显存** | 推理那一刻 GPU 上真正被占用的量 | 权重 + KV Cache + 激活/中间张量 + 框架开销 | **决定能不能跑的是这个** |

```
峰值显存 = 权重
         + KV Cache      (2 × 层数 × KV头数 × head_dim × 字节 × token 数 × 并发数)
         + 激活/中间张量  (随分辨率、音频长度、batch 线性增长)
         + 框架开销      (CUDA context ≈ 0.3-0.5GB；vLLM 默认预分配 90% 显存)
```

**3070 8GB 的可支配显存约 7.2GB**（Windows 桌面合成常驻 0.5–0.8GB），不是 8GB。

---

## 二、核对后的模型矩阵

分级口径（绝对阈值，便于横向比较）：

- **A 开箱即跑** — 峰值 ≤ 4.0GB
- **B 可行需限制** — 4.0–6.2GB，要限分辨率/音频长度/上下文
- **C 边缘** — 6.2–7.2GB，必须量化 + 严格限制，随时可能 OOM
- **D 不可行** — > 7.2GB，或厂商最低要求就超 8GB

| # | 模型 | 任务 | 参数 | BF16 权重 | Q4 权重 | 峰值显存估算 | 3070 | 部署栈 |
|---|---|---|---|---|---|---|---|---|
| 1 | HunyuanOCR | OCR | 1.0B | 2.0GB | 0.6GB | 3.0–3.8GB | **A** | vLLM/Transformers |
| 2 | GLM-OCR | OCR | 0.9B | 1.8GB | 0.55GB | 4.0GB ⚠️官方值 | **A** | vLLM/SGLang/Ollama/MLX |
| 3 | Qwen3-ASR-0.6B | ASR | 0.94B | 1.8GB | 0.5GB | 2.1–3.9GB | **A** | vLLM/transcribe.cpp |
| 4 | TeleOCR | OCR | 1.2B | 2.4GB | 0.7GB | 4.0–6.0GB | **B** | vLLM(async) |
| 5 | MiniCPM-OCR 1.0 | OCR | 1.3B | 2.42GB | 1.43GB | 3.0–4.4GB | **B** | vLLM/SGLang/llama.cpp/Ollama |
| 6 | LFM3-1.7B | 文本 | 1.73B | 3.5GB | 1.0GB | 1.0–4.3GB | **B** | Transformers/GGUF/MLX/NVFP4 |
| 7 | LFM3-Audio-2B | ASR+TTS | 2.0B | 4.0GB | 1.1GB | 1.0–5.3GB | **B** | Transformers/GGUF/MLX/NVFP4 |
| 8 | Qwen3.5-2B | 文本+视觉 | 2.27B | 4.0GB | 1.28GB | 2.2–5.7GB | **B** | vLLM/SGLang/llama.cpp/Ollama |
| 9 | GLM-ASR-Nano-2512 | ASR | 1.5B | 3.5GB | 1.0GB | 4.5–5.9GB | **B** | Transformers/SGLang |
| 10 | Cohere Transcribe 03-2026 | ASR | 2.1B | 4.2GB | 1.21GB | 4.6–6.0GB | **B** | Transformers/vLLM/GGUF |
| 11 | Qwen3-VL-Embedding-2B | Embedding | 2.1B | 4.4GB | 1.1GB | 4.5–6.2GB | **B** | vLLM ≥ 0.14 |
| 12 | Qwen3-ASR-1.7B | ASR | 2.3B | 3.9GB | 1.2GB | 4.0–6.4GB | **C** | vLLM/Transformers |
| 13 | Granite Speech 4.1 2B | ASR/AST | 2.3B | 4.3GB | 2.14GB | 3.0–6.4GB | **C** | vLLM/Transformers/GGUF |
| 14 | FireRed-OCR | OCR | 2.13B | 4.3GB | 1.3GB | 6.0–7.2GB | **C** | Transformers/vLLM |
| 15 | VoxCPM2 | TTS | 2.3B | 5.0GB | 1.1GB | 7.3–8.0GB | **D** | voxcpm 库/ONNX |
| 16 | OXR-1.0 | OCR | 1.3B | 3.2GB | 1.0GB | 24GB+ ⚠️官方要求 | **D** | 专用栈（无 llama.cpp） |

> 机器可读版本：[`data/models.csv`](data/models.csv)（含 `evidence` 证据来源列与 `hard_min_vram_gb` 厂商硬性要求）
> 上表数字来自 `python scripts/vram_estimate.py --csv data/models.csv --ctx 8192`，与该脚本输出逐行一致。

⚠️ = 该行有厂商明确给出的最低显存，优先级高于公式估算。

---

## 三、原表的 5 处硬伤（已修正）

### 1. 「极充裕（GGUF Q8 约 1.7GB）」低估了实际占用

MiniCPM-OCR 的 1.7GB 是 **LLM Q8_0（0.812GB）+ 视觉投影 mmproj（0.901GB）** 两块相加——投影层占了整整一半。
llama.cpp 只传 `--model` 不传 `--mmproj` 会直接报错，很多人卡在这一步。
而且这仍然只是权重，不含 KV 与视觉激活。真实峰值在 3–4.4GB，**它的 A 档不成立，是 B 档**。

```bash
# 必须两个文件一起加载
llama-server --model MiniCPM-OCR-1_0-Q8_0.gguf --mmproj mmproj-model-f16-4x.gguf \
  --ctx-size 8192 --n-gpu-layers 999
```

### 2. FireRed-OCR 不是 1.3B，是 2.13B

它基于 Qwen3-VL-2B-Instruct。BF16 权重 4.3GB，加上视觉 token 激活达到 6.0–7.2GB，
3070 上属于 **C 档**，必须降分辨率或换 FP8/Q4。

### 3. GLM-OCR 官方自己标了 4GB 运行显存

不是"极充裕"。0.9B 参数量确实小，但 CogViT 视觉编码器吃高分辨率图像，官方给的 4GB 才是可用数字。

### 4. OXR 根本不在 8GB 卡的讨论范围内

官方 README 要求 **24GB 起、推荐 48GB**，且**没有 llama.cpp / Ollama 路径**。
它权重只有 3.2GB——这是"权重小 ≠ 能跑"最典型的反例，也是原表最危险的一条。

### 5. VoxCPM2 实测峰值 7.3–7.9GB

官方"建议 8GB"是门槛不是舒适区，实测短句就已经顶到 7.9GB，而 3070 可用只有 7.2GB。
**BF16 档在 3070 上会 OOM**，必须走 q8（2.3GB）或 4-bit（1.1GB）。

### 补充发现（原表未提及）

- **Qwen3.5-2B 是原生多模态模型**（自带视觉编码器），且有 **262K 上下文**。真瓶颈不是权重而是 KV Cache。
- **LFM3 系列不是标准开源**：LFM Open License v1.0 有年营收 1000 万美元上限，商用前必须看许可。
- **Cohere Transcribe 需要外部指定语言**，模型不做语种检测，也没有时间戳/说话人对齐。
- **Granite Speech 官方给了完整显存表**（BF16 6GB / Q8 4GB / Q4_K_M 3GB），是最透明的厂商之一。

---

## 四、部署栈选择：3070 上最容易翻车的地方

| 栈 | 显存行为 | 适合 8GB 卡？ |
|---|---|---|
| **llama.cpp / Ollama** | 按需分配，`n-gpu-layers` 可只卸载部分层到 GPU | ✅ 首选。8GB 上跑 2B 类模型最稳 |
| **Transformers** | 权重 + 激活按需，无预分配 | ✅ 可以，注意 `max_memory` 与 `device_map` |
| **vLLM / SGLang** | `gpu_memory_utilization` 默认 **0.9**，启动就把 90% 显存划进 KV 池 | ⚠️ 必须显式改，否则 0.6B 模型也可能 OOM |

vLLM 在 8GB 卡上的最小可用启动参数：

```bash
vllm serve <model> \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --enforce-eager          # 省掉 CUDA graph 的额外显存
```

---

## 五、研究路线

### Phase 0 — 基建（1 天）
- [ ] 跑 `scripts/vram_estimate.py`，生成基线表存档
- [ ] 固定环境：驱动 / CUDA / torch 版本、`nvidia-smi` 空载基线
- [ ] 统一测量：`torch.cuda.max_memory_allocated()` + `nvidia-smi` 双记录，取峰值

### Phase 1 — 纯文本 + Embedding 打底（2–3 天）
Qwen3.5-2B → LFM3-1.7B → Qwen3-VL-Embedding-2B
- 目标：拿到干净的「参数量 → 峰值显存」基线，量化估算公式误差
- 变量：量化档位（BF16/Q8/Q4）× 上下文（2K/8K/32K）× batch

### Phase 2 — 多模态 OCR 主战场（1 周）
MiniCPM-OCR → GLM-OCR → HunyuanOCR → TeleOCR → FireRed-OCR
- 目标：画出「输入分辨率 × 视觉 token 数 × 峰值显存」曲线
- 指标：单页 PDF 耗时、峰值显存、Markdown 结构正确率
- OXR 只做云端 API 对照，不在本机跑

### Phase 3 — 语音（1 周）
Qwen3-ASR-0.6B → Granite Speech → Cohere Transcribe → GLM-ASR → LFM3-Audio → VoxCPM2
- 目标：找「音频时长 → 峰值显存」的拐点
- VoxCPM2 单独做：4-bit → q8 → BF16 逐档压测，记录 OOM 边界

### Phase 4 — 产出
- [ ] `bench/<model>.md`：环境、命令、峰值显存、速度、失败案例
- [ ] 一张「输入规模 vs 显存」总图
- [ ] 一份《8GB 卡选型建议》

---

## 六、复现

```bash
git clone https://github.com/<your-name>/2B.git
cd 2B
python scripts/vram_estimate.py --csv data/models.csv --gpu-mem 8 --ctx 8192
```

只看单个模型、换上下文、换精度：

```bash
python scripts/vram_estimate.py --model qwen3.5-2b --ctx 32768
python scripts/vram_estimate.py --csv data/models.csv --precision q4
```

脚本只做**静态估算**（不加载权重），用于在下载几十 GB 之前先判断可行性。

---

## 七、数据来源与可信度

`data/models.csv` 的 `evidence` 列分三级：

| 标记 | 含义 | 可信度 |
|---|---|---|
| 官方文件体积 / 官方文档 / 官方 README | 直接从 HF 仓库文件大小或厂商文档读到 | 高 |
| 第三方实测 | 独立评测站点的实测峰值 | 中 |
| 自行推算 | 按公式从参数量推导 | 低，必须实测校准 |

**所有 `peak_vram_est_gb` 都是估算，不是保证。** Phase 1 的首要任务就是用实测把这张表校准一遍。

---

## License

文档部分 CC BY 4.0。各模型权重遵循其原始许可（见 `data/models.csv` 的 `license` 列）。
