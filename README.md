# Gemma-4-E2B-Chinese-R1 🧠✨

基于 **Google Gemma 4 (E2B)** 原生轻量基座，在 **Google Colab (Tesla T4 GPU)** 上使用 **Unsloth + QLoRA (4-bit)** 完成的**通用中文指令 + DeepSeek-R1 深度思考链（CoT）**双融合微调项目。

包含端到端训练 Notebook，以及针对 **Apple Silicon (Mac M系列 MPS 加速) / NVIDIA CUDA** 深度优化的流式交互终端对话脚本 `chat.py`。

[![Base Model](https://img.shields.io/badge/Base%20Model-Google%20Gemma%204%20E2B-blue)](https://huggingface.co/google)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA%204--bit%20(Unsloth)-green)](https://github.com/unslothai/unsloth)
[![Alpaca Dataset](https://img.shields.io/badge/Dataset-alpaca--zh%20(2k)-orange)](https://huggingface.co/datasets/shibing624/alpaca-zh)
[![R1 Dataset](https://img.shields.io/badge/Dataset-DeepSeek--R1--Distill%20(1k)-purple)](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

---

## 🌟 项目特色 (Highlights)

* **极低算力成本**：在 Google Colab 免费级 **Tesla T4 (15GB)** 单卡上，通过 Unsloth 4-bit 量化仅耗时约 **35 分钟** 即可完成微调，极度节省算力单元。
* **双模态能力融合**：
  * **2,000 条** `alpaca-zh`：赋予基座规范、流畅的日常中文问答与指令遵循能力；
  * **1,000 条** `Chinese-DeepSeek-R1-Distill`：注入 DeepSeek-R1 `<think>` 慢思考灵魂，支持逻辑反思与逐步推导。
* **开箱即用的终端交互 (`chat.py`)**：
  * ⚡ **自动硬件加速**：原生优先调用 Apple Silicon 的 **Metal (MPS)** GPU 加速，同时支持 NVIDIA CUDA 与 CPU 优雅降级；
  * 🌊 **实时打字机流式输出**：基于 `TextStreamer` 逐字吐出推理与回答；
  * 🧠 **思维链管理与历史隔离**：实时展示思考过程，在多轮对话中自动剥离思考痕迹，仅将最终回答存入上下文，防止历史窗口污染与漂移。

---

## ⚙️ 实验与微调配置 (Configuration)

| 维度 | 配置详情 |
| :--- | :--- |
| **计算平台** | Google Colab (通过 VS Code 远程连接) |
| **计算卡** | NVIDIA Tesla T4 (15.36 GB 显存) |
| **基础底座** | `unsloth/gemma-4-E2B-it` (4-bit 预量化版本) |
| **微调算法** | QLoRA (Rank $r=16$, $\alpha=16$, dropout=0) |
| **LoRA 目标模块** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **融合数据集** | `alpaca-zh` (2000条) + `Chinese-DeepSeek-R1-Distill-110k` (1000条过滤精选) |
| **总数据量** | 2,778 条有效混合样本 |
| **序列长度** | `max_seq_length = 2048` |
| **Batch 大小** | `per_device_batch_size = 2`, `gradient_accumulation_steps = 4` (等效 Batch = 8) |
| **训练步数** | 1 个完整 Epoch (~350 steps), `lr = 2e-4`, 余弦退火 |
| **单卡训练耗时** | 约 **30 ~ 35 分钟** (T4 显卡) |

---

## 📁 目录结构 (Directory Structure)

```text
mygemma/
├── gemma.ipynb              # 端到端 Colab 微调训练 Notebook (Unsloth + 数据融合)
├── gemma_tpu.ipynb          # TPU 实验测试 Notebook
├── chat.py                  # 本地多轮流式对话客户端 (支持 MPS/CUDA, R1 思考链渲染)
├── requirements.txt         # 核心运行依赖
├── README.md                # 项目文档
├── my_gemma_lora/           # 导出的通用微调 LoRA 权重 (~92MB)
└── my_gemma_r1_lora/        # 导出的 R1 思维链微调 LoRA 权重 (~92MB)
```

---

## 🚀 快速上手 (Quick Start)

### 1. 本地环境准备

推荐使用项目专用的 Python 3.11+ 虚拟环境：

```bash
# 1. 克隆/进入项目目录
cd mygemma

# 2. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# 3. 安装依赖包
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 启动终端智能对话

直接运行 `chat.py` 即可开启交互对话（首次启动会自动下载 Gemma 底座模型，之后秒开）：

```bash
python chat.py
```

* **输入退出**：键入 `exit` 或 `q`
* **清空历史**：键入 `clear`

终端效果示例：

```text
============================================================
🚀 正在启动专属微调模型：Gemma-4-E2B（可思考版）
============================================================
⚡️ 检测到 Apple Silicon，已启用 MPS GPU 加速！
📦 正在加载分词器与基座模型...
🧩 正在挂载专属 LoRA 微调适配器...
🎉 模型就绪！输入问题后回车发送。
============================================================

🧑 你: 9.11 和 9.9 哪个更大？
🤖 Gemma: <|channel>thought
Thinking Process:
1. Analyze the two numbers: 9.11 and 9.9.
2. Compare the whole number part: both are 9.
3. Compare the tenths place: 9.11 has 1, 9.9 has 9.
4. Since 9 > 1, 9.9 is larger.
<channel|>9.9 更大。因为在十分位上，9.9 是 9，而 9.11 是 1，所以 9.9 大于 9.11。
```

---

## 🛠️ 云端训练指南 (Training in Colab)

所有训练逻辑均在 `gemma.ipynb` 中实现：

1. **安装 Unsloth 加速库**：
   ```bash
   pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

2. **数据载入与标准模板映射**：
   使用标准对话模板 `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...` 格式化训练样本，并将思维链自然包裹于 `<think>` 标签内。

3. **模型保存与下载**：
   训练结束后执行：
   ```python
   model.save_pretrained("my_gemma_r1_lora")
   tokenizer.save_pretrained("my_gemma_r1_lora")
   ```
   下载文件夹后放置于本地项目根目录即可直接由 `chat.py` 挂载。

---

## 💡 深度推理优化与踩坑总结 (Troubleshooting)

### 1. 推理过程截断，没有输出最终答案？
* **原因**：R1 蒸馏模型在给出结论前会进行长篇幅的思考链展开，默认的 `max_new_tokens=512` 会在思考未完成时耗尽 Token。
* **解决**：在 `chat.py` 中将 `max_new_tokens` 提升至 `2048` 或 `4096`。

### 2. `AttributeError: KeyError: 'shape'` 报错？
* **原因**：新版 `transformers` (v5.x) 中 `apply_chat_template` 返回的是 `BatchEncoding` 字典对象，直接传入 `model.generate()` 会因无法解析 `.shape` 报错。
* **解决**：代码中通过 `hasattr(encoded, "input_ids")` 显式解包为 Tensor 后再送入模型。

### 3. 小参数模型（2B）推理效果与建议
* **思考触发不稳定的现象**：2B 极小参数模型在某些直觉简单题上可能偷懒跳过思考通道；
* **中英跨语言漂移**：纯英文训练思维链容易在“打折”、“成语”等中文语境产生误解（如将“打八折”理解为“80% discount/两折”）；
* **进阶选型建议**：如果算力有限但追求极高中文推理质量，推荐使用 **Qwen2.5-3B** 或 **Qwen2.5-7B** 作为底座，配合 500~1000 条高质量中文 R1 思维链数据微调，中文表现将大幅跃升。

---

## 🤝 致谢与开源协议 (License)

* 基础模型遵循 Google [Gemma Terms of Use](https://ai.google.dev/gemma/terms)；
* 训练加速与量化由 [Unsloth AI](https://github.com/unslothai/unsloth) 提供强力支持；
* 训练语料来源于开源社区 [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) 与 [Chinese-DeepSeek-R1-Distill-data-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT)；
* 本项目代码采用 [Apache 2.0 开源许可证](LICENSE)。
