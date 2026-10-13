# Qwen2.5-3B & Gemma 4 深度思考微调实战 (DeepSeek-R1 CoT Distillation)

本项目记录了在低算力资源约束（**Google Colab Tesla T4 15GB 单卡**）下，使用 **Unsloth + QLoRA (4-bit)** 框架，对轻量级开源大模型（**Qwen2.5-3B** 与 **Gemma 4-E2B**）进行 **DeepSeek-R1 纯中文深度思维链（CoT）慢思考蒸馏** 的完整实战全流程。

包含 20 分钟极速训练 Notebook、本地 Apple Silicon (Metal MPS) GPU 流式推理交互终端，以及真实复杂的逻辑与业务场景评测记录。

[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--3B%20%7C%20Gemma%204-blue)](https://huggingface.co/Qwen)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA%204--bit%20(Unsloth)-green)](https://github.com/unslothai/unsloth)
[![Dataset](https://img.shields.io/badge/Dataset-Chinese--DeepSeek--R1--Distill-orange)](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT)
[![Hardware](https://img.shields.io/badge/Inference-Apple%20Silicon%20MPS-black)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

---

## 📖 项目亮点 (Highlights)

* **极致算力性价比**：
  * 基于 `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` 预量化底座；
  * 仅需消费级/免费级 **Tesla T4 (15GB)** 显卡，显存占用仅 **~3.5GB**；
  * 全流程微调仅需 **18 ~ 22 分钟**，消耗不到 1 个计算单元（Compute Unit）。
* **纯血中文思维链对齐**：
  * 精选 1,000 条严格闭环的中文 DeepSeek-R1 `<think>...</think>` 推理数据；
  * 彻底解决轻量级模型“中英思维漂移”与“偷懒跳过思考通道”两大硬伤。
* **干净轻量的 Mac 本地推理客户端 (`chat.py`)**：
  * 原生启用 Apple Silicon **Metal (MPS)** GPU 加速，秒级启动；
  * 自动管理多轮对话，流式输出思考过程，并在存入历史前自动剥离思考痕迹；
  * 针对 3B 小模型调优低温度（`temperature = 0.2`），有效压制过度反思综合征。

---

## ⚡ 实测推理战报 (Benchmark & Results)

在本地通过 `chat.py` 对微调后的 **Qwen2.5-3B-R1** 进行了 4 道经典逻辑、数学陷阱与实际业务问题的实测：

| 测试题目 | 题目类型 | 思考表现 | 测试结果 |
| :--- | :--- | :--- | :---: |
| **打折陷阱**：一件商品先降价 20%，随后又涨价 20%，现价比原价高还是低？相差百分之几？ | 中文商业数学 | `<think>` 内清晰推导：100 → 80 → 96，差额 4 元，无过度怀疑，果断收敛。 | **通过 (4%)** |
| **工程并发效率**：3 个人挖 3 个坑需要 3 天。9 个人挖 9 个坑需要多少天？ | 归一/并发逻辑 | 识破直觉陷阱，在思考链中指明“人数与任务等比例增加，单人单坑耗时不变”。 | **通过 (3天)** |
| **仓储机器人调度**：A 车电量 85% 距桩 10 米，B 车电量 12% 距桩 60 米，单桩空闲选谁？ | 工业调度运筹 | 准确识别低电量引发死锁断电的业务风险，分析多维度约束，决策逻辑严密。 | **通过 (优先B车)** |
| **恒定年龄差**：我 6 岁时妹妹是我的一半。当我 70 岁时妹妹几岁？ | 语言歧义陷阱 | 误将“我的一半”当成了固定绑定函数（70÷2=35），属于小模型典型字面理解盲区。 | 未通过 (35岁) |

> **实测总评**：在仅微调 1,000 条数据、耗时 20 分钟的前提下，3B 小模型取得了 **75% 的高难度陷阱题命中率**。

---

## 📁 目录结构 (Directory Structure)

```text
mygemma/
├── qwen_train.ipynb         # [推荐] Qwen2.5-3B + R1 思考微调 Notebook (Colab 20分钟跑通)
├── gemma.ipynb              # Gemma 4-E2B 前期微调探索 Notebook
├── chat.py                  # 本地多轮流式对话客户端 (支持 MPS 加速 / 纯净文本界面)
├── requirements.txt         # 核心运行依赖
├── README.md                # 项目全景技术文档
├── my_qwen_r1_lora/         # 导出的 Qwen2.5-3B LoRA 配置与分词器 (权重已通过 gitignore 规避)
└── my_gemma_lora/           # 导出的 Gemma 4 LoRA 适配器权重
```

---

## 🚀 本地运行体验 (Local Quick Start)

### 1. 环境准备

```bash
# 进入项目目录
cd mygemma

# 创建并激活专用虚拟环境
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows

# 安装核心依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 启动流式对话客户端

确保 `my_qwen_r1_lora` 权重目录位于项目根目录下，直接执行：

```bash
python chat.py
```

* **退出对话**：输入 `exit` 或 `q`
* **清空历史**：输入 `clear`

**终端运行效果预览**：

```text
============================================================
启动专属微调推理模型: Qwen2.5-3B (DeepSeek-R1 思考版)
============================================================
[硬件] 检测到 Apple Silicon，已启用 Metal (MPS) GPU 加速
[加载] 正在加载分词器: /path/to/my_qwen_r1_lora
[加载] 正在加载 Qwen2.5-3B 底座权重 (Qwen/Qwen2.5-3B-Instruct)...
[挂载] 正在注入 LoRA 适配器: /path/to/my_qwen_r1_lora

模型加载完成。输入问题后回车发送。
提示: 输入 'exit' 或 'q' 退出；输入 'clear' 清空对话历史。
============================================================

你: 3 个人挖 3 个坑需要 3 天。照这个工作速度，9 个人挖 9 个坑需要多少天？
Qwen: <think>
题目中的条件是“3个人挖3个坑需要3天”。这里的关键点在于每个人的工作时间不变，无论人数如何变化。
每个人挖一个坑需要3天，当增加到9个人时，每个人仍然需要3天完成自己的任务，而9个人同时工作，每个人的工作时间不变。
</think>
总的工作时间仍然是每个人的工作时间，即 3 天。
```

---

## 🛠️ 云端微调实战指南 (Training on Colab)

所有训练逻辑均封装在 [`qwen_train.ipynb`](file:///Users/mac/IT/mygemma/qwen_train.ipynb) 中，仅需 3 步：

1. **上传 Notebook**：打开 [Google Colab](https://colab.research.google.com/)，上传 `qwen_train.ipynb`；
2. **选择 GPU 规格**：运行时设置硬件加速器为 **T4 GPU**；
3. **点击全部运行**：
   * 自动安装 Unsloth 极速框架；
   * 自动加载预量化 4-bit 底座；
   * 自动过滤 1,000 条纯中文 R1 思维链数据；
   * 训练 1 个 Epoch（约 18 分钟）；
   * 现场自动测试开炉效果，并自动将权重打包为 `my_qwen_r1_lora.zip` 下载至本地。
4. **断开释放**：下载完成后，及时点击 `代码执行程序 -> 断开连接并删除运行时` 即可锁住算力。

---

## 💡 核心经验与避坑实录 (Lessons Learned)

### 1. 底座语料分布的重要性（Gemma vs Qwen）
* **Gemma 4** 英文语料占绝大多数，在中文推理中容易出现“先在心里翻译为英文再思考”的跨语言偏差（如将中文特有的“打八折”误翻为“80% discount / 打两折”）；
* **Qwen2.5** 拥有深厚的原生中文预训练语料沉淀，在面对中文成语、商业习惯及多步归一数学题时表现极为稳健。

### 2. 小参数模型（3B）的“反思退化 / 过度内耗”
* **现象**：3B 级别模型学习了 R1 思维链中的“自我怀疑（Wait, let me double check）”模式，但在较高温度采样下容易在初次算对后，脑补各种不存在的限制，从而推翻正确答案；
* **解法**：在推理阶段将采样温度收紧至 `temperature = 0.2`（或使用贪婪采样），能有效促使模型在首次严密推导后迅速收敛并输出结论。

### 3. GitHub 100MB 单文件限制
* 导出的 `adapter_model.safetensors` 文件体积约 114MB，超过了 GitHub 普通仓库单文件 100MB 的上限；
* 本项目通过在 [`.gitignore`](file:///Users/mac/IT/mygemma/.gitignore) 中排除该大文件，同时保留完整配置文件、分词器与重现代码，确保仓库能够顺利进行常规 Git 推送。

---

## 🤝 致谢与协议 (License)

* 基础模型由 [Alibaba Qwen 团队](https://github.com/QwenLM/Qwen2.5) 与 [Google Gemma 团队](https://ai.google.dev/gemma) 开源；
* 微调加速由 [Unsloth AI](https://github.com/unslothai/unsloth) 提供强力支持；
* 训练语料来源于 [Chinese-DeepSeek-R1-Distill-data-110k-SFT](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT)；
* 本仓库代码与 Notebook 遵循 [Apache 2.0 开源许可协议](LICENSE)。
