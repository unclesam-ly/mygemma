# Gemma-4-E2B-Chinese-Alpaca 🦙✨

基于 **Google Gemma 4** 原生多模态轻量基座，在 **Google Colab (Tesla T4 GPU)** 上使用 **Unsloth + QLoRA (4-bit)** 完成的中文指令微调项目。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Google%20Gemma%204%20E2B-blue)](https://huggingface.co/google)
[![Fine-Tuning](https://img.shields.io/badge/Method-QLoRA%204--bit%20(Unsloth)-green)](https://github.com/unslothai/unsloth)
[![Dataset](https://img.shields.io/badge/Dataset-alpaca--zh%20(5000%20samples)-orange)](https://huggingface.co/datasets/shibing624/alpaca-zh)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

---

## 📖 项目简介 (Overview)

本项目演示了如何在消费级/免费云端算力（**单卡 Tesla T4 15GB 显存**）上，以极低显存开销与高吞吐速度完成 Google 最新的 **Gemma 4** 架构模型微调：
* **高效微调**：借助 Unsloth 核心优化，将 2B~4B 级别模型显存压缩至 ~4GB，训练提速 2~5 倍；
* **中文适配**：基于经典的开源中文高质量指令集 `shibing624/alpaca-zh`（5,000 条），让英文基座快速具备地道、规范的中文遵循与问答能力；
* **工程友好**：代码兼容 VS Code 远程连接 Colab 环境，自带多模态 Processor 避坑处理与防爆内存导出方案。

---

## ⚙️ 硬件与实验配置 (Configuration)

| 维度 | 配置详情 |
| :--- | :--- |
| **计算平台** | Google Colab（通过 VS Code 远程连接内核） |
| **计算卡** | NVIDIA Tesla T4 (15.36 GB 显存) |
| **基础模型** | `unsloth/gemma-4-E2B-it` (4-bit 预量化版本) |
| **微调算法** | QLoRA (Rank $r=16$, $\alpha=16$, dropout=0) |
| **目标模块** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **训练数据** | `shibing624/alpaca-zh` 前 5,000 条指令样本 |
| **序列长度** | `max_seq_length = 2048` |
| **Batch 大小** | `per_device_batch_size = 2`, `gradient_accumulation_steps = 4` (有效 Batch = 8) |
| **学习率与步数** | `lr = 2e-4`, 余弦退火, `warmup_steps = 10`, 训练 1 个完整 Epoch (~625 steps) |
| **单卡训练耗时** | 约 **31 ~ 37 分钟**（T4 GPU） |

---

## 📁 目录结构 (Directory Structure)

```text
mygemma/
├── gemma.ipynb              # 完整端到端微调 Jupyter Notebook
├── README.md                # 本说明文档
└── my_gemma_lora/           # 训练完成导出的 LoRA 适配器权重（约 50MB~80MB）
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── tokenizer.json
    └── ...
```

---

## 🚀 核心流程与代码解析 (Workflow)

整个微调流程已封装在 `gemma.ipynb` 中，主要分为以下步骤：

### 1. 环境安装
安装适配 Gemma 4 与 CUDA 环境的加速库：
```bash
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. 模型加载与 LoRA 注入
使用 4-bit 精度加载 Gemma 4，配置注意力层与前馈层的低秩分解矩阵：
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

### 3. 数据集载入与 Gemma 官方模版格式化
针对 Gemma 的 `<start_of_turn>` 标记进行模板包装：
```python
from datasets import load_dataset

dataset = load_dataset("shibing624/alpaca-zh", split="train[:5000]")

def format_prompts(batch):
    texts = []
    for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        user_msg = f"{inst}\n{inp}".strip() if inp else inst
        text = f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n{out}<end_of_turn>"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched = True)
```

### 4. 启动微调训练
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = True,
        logging_steps = 10,
        output_dir = "outputs",
    ),
)
trainer.train()
```

### 5. 推理验证（多模态兼容规范）
由于 Gemma 4 属于原生多模态架构，底层采用 `Processor` 处理输入，纯文本推理时需显式传入 `text=[prompt]`：
```python
FastLanguageModel.for_inference(model)

prompt = "<start_of_turn>user\n请用通俗易懂的语言解释什么是分布式系统？<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(text = [prompt], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, temperature = 0.7)
print(tokenizer.batch_decode(outputs)[0])
```

---

## 💾 权重导出与部署 (Deployment)

### 方案 A：保存轻量级 LoRA 权重（推荐）
仅导出增量权重，体积小（约 50MB~80MB），上传与分发极快：
```python
model.save_pretrained("my_gemma_lora")
tokenizer.save_pretrained("my_gemma_lora")
```

**本地 Python 加载使用**：
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
model = PeftModel.from_pretrained(base_model, "./my_gemma_lora")
```

### 方案 B：本地 Ollama 部署
如果通过合并导出了 GGUF 文件（如 `my_gemma_model-unsloth.Q4_K_M.gguf`），可在终端直接运行：
```bash
# 编写 Modelfile
echo "FROM ./my_gemma_model-unsloth.Q4_K_M.gguf" > Modelfile

# 创建并启动 Ollama 本地服务
ollama create my-gemma -f Modelfile
ollama run my-gemma
```

---

## 🛠️ 常见避坑记录 (Troubleshooting)

1. **`Unsloth cannot find any torch accelerator?`**
   * **原因**：VS Code 或 Colab 当前连接的是 CPU 运行时。
   * **解决**：在运行时设置中将硬件加速器切换为 **T4 GPU**。
2. **`Gemma 4 E4B 内存溢出 (Kernel 崩溃 / SIGKILL: 9)`**
   * **原因**：T4 显卡不支持硬件级 `bfloat16`，自动提升为 `float32` 导致显存和系统内存需求翻倍；加之 Gemma 4 的 25.6 万巨量词表解压，顶破了 Colab 12GB RAM 限制。
   * **解决**：改用轻量紧凑的 **Gemma 4 E2B** 或 **Gemma 2 2B**，显存仅占 ~4GB，完全不爆内存。
3. **`prepare_inputs_layout` 报错**：
   * **原因**：Gemma 4 是多模态模型，`tokenizer([prompt])` 默认第一项为图像输入。
   * **解决**：显式传入命名参数 `tokenizer(text = [prompt], return_tensors = "pt")`。

---

## 🤝 致谢与协议 (License)

* 基座模型遵循 Google [Gemma Open Model License](https://ai.google.dev/gemma/terms)（允许免费学术与商业使用）；
* 微调加速框架来自 [Unsloth](https://github.com/unslothai/unsloth)；
* 训练数据集基于 [alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)；
* 本项目的代码与微调权重遵循 [Apache 2.0 开源许可协议](LICENSE)。

