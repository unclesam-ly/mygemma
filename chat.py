import os
import sys
import torch
from transformers import TextStreamer

# ── 1. 选择加速设备 ────────────────────────────────────────────────
def pick_device():
    if torch.backends.mps.is_available():
        print("⚡️ 检测到 Apple Silicon，已启用 MPS GPU 加速！")
        return torch.device("mps"), torch.float16
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("⚡️ 已启用 NVIDIA CUDA GPU 加速！")
        return torch.device("cuda"), dtype
    print("⚠️  未检测到 GPU，使用 CPU 运行（会比较慢）。")
    return torch.device("cpu"), torch.float32


def main():
    print("=" * 60)
    print("🚀 正在启动专属微调模型：Gemma-4-E2B（可思考版）")
    print("=" * 60)

    device, dtype = pick_device()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lora_path   = os.path.join(current_dir, "my_gemma_lora")

    if not os.path.exists(lora_path):
        print(f"❌ 未找到 LoRA 权重目录: {lora_path}")
        sys.exit(1)

    # ── 2. 用 unsloth 加载（底座是 4-bit 量化版，必须用 unsloth） ──
    print("📦 正在加载模型与分词器（首次需要下载底座权重约 1.3GB）...")
    try:
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            lora_path,
            max_seq_length = 8192,    # 上下文窗口
            load_in_4bit   = True,    # 与训练时保持一致
        )
        FastModel.for_inference(model)  # 开启推理加速
        print("✅ unsloth FastModel 加载成功！")
        use_unsloth = True
    except ImportError:
        print("⚠️  未安装 unsloth，回退到标准 transformers 加载...")
        use_unsloth = False

    if not use_unsloth:
        # 回退方案：用标准 peft + transformers（速度较慢）
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        tokenizer  = AutoTokenizer.from_pretrained(lora_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/gemma-4-E2B-it",   # 非量化公开版底座
            torch_dtype       = dtype,
            low_cpu_mem_usage = True,
        )
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.to(device)
        model.eval()

    # ── 3. 流式输出器 ─────────────────────────────────────────────
    actual_tok = getattr(tokenizer, "tokenizer", tokenizer)
    streamer   = TextStreamer(actual_tok, skip_prompt=True, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print("🎉 模型就绪！输入问题后回车发送。")
    print("💡 输入 'exit' / 'q' 退出；输入 'clear' 清空对话历史。")
    print("=" * 60 + "\n")

    history = []   # [{"role": "user"/"assistant", "content": "..."}]

    while True:
        try:
            user_input = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("👋 再见！")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("🧹 对话历史已清空。\n")
            continue

        history.append({"role": "user", "content": user_input})

        # ── 4. 用 chat_template 构造 prompt（含 enable_thinking） ──
        # apply_chat_template 在新版 transformers 返回 BatchEncoding（字典），
        # 需要显式取出 input_ids tensor 才能传给 generate()
        try:
            encoded = tokenizer.apply_chat_template(
                history,
                tokenize              = True,
                add_generation_prompt = True,
                return_tensors        = "pt",
                enable_thinking       = True,   # 🧠 启用推理链
            )
        except TypeError:
            # 旧版 transformers 不支持 enable_thinking 参数，降级处理
            encoded = tokenizer.apply_chat_template(
                history,
                tokenize              = True,
                add_generation_prompt = True,
                return_tensors        = "pt",
            )

        # 兼容返回纯 tensor 或 BatchEncoding 两种情况
        if hasattr(encoded, "input_ids"):
            prompt_ids = encoded.input_ids
        else:
            prompt_ids = encoded

        target_device = model.device if hasattr(model, "device") else device
        prompt_ids    = prompt_ids.to(target_device)
        prompt_len    = prompt_ids.shape[-1]

        # ── 5. 推理生成 ───────────────────────────────────────────
        print("🤖 Gemma: ", end="", flush=True)
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids,
                streamer       = streamer,
                max_new_tokens = 4096,    # 思考链需要较多 token
                temperature    = 0.6,     # 推理模型建议低温
                top_p          = 0.9,
                do_sample      = True,
                pad_token_id   = actual_tok.eos_token_id,
            )
        print()

        # ── 6. 提取模型回复，加入历史（剥离思考链，只保留最终答案） ──
        new_ids    = output_ids[0][prompt_len:]
        full_reply = actual_tok.decode(new_ids, skip_special_tokens=False)

        # 剥离 <|channel|>thought\n...<turn|> 思考块，只把最终答案存入历史
        import re
        clean_reply = re.sub(
            r"<\|channel\|>thought\n.*?<turn\|>",
            "",
            full_reply,
            flags=re.DOTALL,
        ).strip()
        # 兼容 <think>...</think> 格式
        clean_reply = re.sub(
            r"<think>.*?</think>",
            "",
            clean_reply,
            flags=re.DOTALL,
        ).strip()

        if clean_reply:
            history.append({"role": "assistant", "content": clean_reply})


if __name__ == "__main__":
    main()
