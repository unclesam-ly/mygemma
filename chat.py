import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

class StopOnTurn(StoppingCriteria):
    def __init__(self, tokenizer, stop_words, prompt_len):
        super().__init__()
        self.tok = getattr(tokenizer, "tokenizer", tokenizer)
        self.stop_words = stop_words
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        # 核心：必须排除前面的 prompt，只检测模型新吐出来的 Token！
        new_tokens = input_ids[0][self.prompt_len:]
        if len(new_tokens) == 0:
            return False
        last_text = self.tok.decode(new_tokens[-10:], skip_special_tokens=False)
        return any(w in last_text for w in self.stop_words)



def main():
    print("=" * 60)
    print("🚀 正在启动专属微调模型：Gemma-4-E2B-Chinese-Alpaca")
    print("=" * 60)

    # 1. 自动选择最佳加速硬件 (优先使用 Apple Silicon M2 Metal GPU 加速)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        print("⚡️ 检测到 Apple Silicon M 系列芯片，已启用 Metal (MPS) GPU 加速！")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("⚡️ 已启用 NVIDIA CUDA GPU 加速！")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("⚠️ 未检测到 GPU，使用 CPU 运行。")

    # 2. 模型与适配器路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lora_path = os.path.join(current_dir, "my_gemma_lora")
    base_model_id = "unsloth/gemma-4-E2B-it"

    if not os.path.exists(lora_path):
        print(f"❌ 未找到 LoRA 权重目录: {lora_path}")
        print("请确保已将 my_gemma_lora 文件夹解压放在当前目录下。")
        sys.exit(1)

    print(f"📦 正在加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
    except Exception:
        tokenizer = AutoProcessor.from_pretrained(lora_path)

    print(f"📥 正在加载基座模型 ({base_model_id})...")
    print("   (首次运行会自动下载官方底座权重，大约 3.5GB，之后将直接从本地缓存秒开)")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype = dtype,
            low_cpu_mem_usage = True,
        )
    except Exception as e:
        print(f"尝试加载时发生错误: {e}")
        print("正在尝试通用多模态架构兼容加载...")
        from transformers import AutoModelForImageTextToText
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype = dtype,
            low_cpu_mem_usage = True,
        )

    print(f"🧩 正在挂载专属 LoRA 微调适配器 ({lora_path})...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(device)
    model.eval()

    # 3. 创建实时流式打字器 (像 ChatGPT / Ollama 一样逐字吐出回答)
    streamer = TextStreamer(
        getattr(tokenizer, "tokenizer", tokenizer),
        skip_prompt = True,
        skip_special_tokens = True,
    )

    print("\n" + "=" * 60)
    print("🎉 模型加载完成！随时可以开始对话。")
    print("💡 提示：输入内容后按回车发送；输入 'exit' 或 'q' 退出程序。")
    print("=" * 60 + "\n")

    # 提取 Gemma 专用的对话结束标记 ID (<end_of_turn> 和 <eos>)，装上“刹车片”
    actual_tok = getattr(tokenizer, "tokenizer", tokenizer)
    stop_token_ids = [actual_tok.eos_token_id]
    end_id = actual_tok.convert_tokens_to_ids("<end_of_turn>")
    if end_id is not None and end_id != actual_tok.unk_token_id:
        stop_token_ids.append(end_id)
    stop_token_ids = list(set([tid for tid in stop_token_ids if tid is not None]))

    while True:
        try:
            user_input = input("🧑 你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 再见！")
                break

            prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"

            # 兼容多模态 Processor 与纯文本 Tokenizer
            if hasattr(tokenizer, "tokenizer"):
                inputs = tokenizer(text = [prompt], return_tensors = "pt").to(device)
            else:
                inputs = tokenizer([prompt], return_tensors = "pt").to(device)

            # 动态终止监听器（只监控新生成的 token，忽略 prompt 前缀）
            prompt_len = inputs.input_ids.shape[1]
            stop_criteria = StoppingCriteriaList([
                StopOnTurn(tokenizer, ["<end_of_turn>", "<turn|>"], prompt_len)
            ])

            print("🤖 Gemma: ", end = "", flush = True)
            with torch.no_grad():
                model.generate(
                    **inputs,
                    streamer = streamer,
                    max_new_tokens = 512,
                    temperature = 0.7,
                    top_p = 0.9,
                    do_sample = True,
                    eos_token_id = stop_token_ids,
                    stopping_criteria = stop_criteria,
                )
            print()

        except KeyboardInterrupt:
            print("\n👋 已中断对话。")
            break
        except Exception as e:
            print(f"\n⚠️ 生成过程中发生异常: {e}\n")

if __name__ == "__main__":
    main()
