import os
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

SYSTEM_PROMPT = "你是一个具备深度逻辑推演能力的助手。在给出最终结论前，请先在 <think> 与 </think> 标签内进行严密的逐步推导与自我审视。若初次推导逻辑完整成立，请直接收敛输出最终结论。"

# ── 1. 选择加速硬件 ────────────────────────────────────────────────
def pick_device():
    if torch.backends.mps.is_available():
        print("[硬件] 检测到 Apple Silicon，已启用 Metal (MPS) GPU 加速")
        return torch.device("mps"), torch.float16
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print("[硬件] 已启用 NVIDIA CUDA GPU 加速")
        return torch.device("cuda"), dtype
    print("[硬件] 未检测到可用 GPU，使用 CPU 运行（速度较慢）")
    return torch.device("cpu"), torch.float32


def main():
    print("=" * 60)
    print("启动专属微调推理模型: Qwen2.5-3B (DeepSeek-R1 思考版)")
    print("=" * 60)

    device, dtype = pick_device()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lora_path   = os.path.join(current_dir, "my_qwen_r1_lora")
    base_model_id = "Qwen/Qwen2.5-3B-Instruct"

    if not os.path.exists(lora_path):
        print(f"[错误] 未找到 LoRA 权重目录: {lora_path}")
        print("请确认已将训练导出的 my_qwen_r1_lora 文件夹放置于当前目录下。")
        sys.exit(1)

    print(f"[加载] 正在加载分词器: {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    print(f"[加载] 正在加载 Qwen2.5-3B 底座权重 ({base_model_id})...")
    print("       (首次加载会自动下载底座，之后直接从本地缓存秒开)")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype       = dtype,
        low_cpu_mem_usage = True,
    )

    print(f"[挂载] 正在注入 LoRA 适配器: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.to(device)
    model.eval()

    # ── 2. 流式输出打字机 ─────────────────────────────────────────
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    print("\n" + "=" * 60)
    print("模型加载完成。输入问题后回车发送。")
    print("提示: 输入 'exit' 或 'q' 退出；输入 'clear' 清空对话历史。")
    print("=" * 60 + "\n")

    history = []   # [{"role": "user"/"assistant", "content": "..."}]

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出对话。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见！")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[信息] 对话历史已清空。\n")
            continue

        history.append({"role": "user", "content": user_input})

        # ── 3. 构造标准 ChatML 对话格式并引导 <think> ──────────────
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = True,
        )
        
        # 强制引导进入思考链
        if not prompt_text.endswith("<think>\n"):
            prompt_text += "<think>\n"

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[-1]

        # ── 4. 推理生成 (温度设为 0.2，避免 3B 小模型过度自我怀疑) ──
        print("Qwen: <think>\n", end="", flush=True)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                streamer       = streamer,
                max_new_tokens = 2048,
                temperature    = 0.2,
                top_p          = 0.95,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )
        print()

        # ── 5. 提取最终结论加入多轮历史 (剥离思考链，避免污染) ──────
        new_ids    = output_ids[0][prompt_len:]
        full_reply = tokenizer.decode(new_ids, skip_special_tokens=False)

        # 剥离 <think>...</think>，只把最终答案存入多轮历史
        clean_reply = re.sub(
            r"<think>.*?</think>",
            "",
            full_reply,
            flags=re.DOTALL,
        )
        # 移除可能残留的 <|im_end|> 等标记
        clean_reply = clean_reply.replace("<|im_end|>", "").strip()

        if clean_reply:
            history.append({"role": "assistant", "content": clean_reply})


if __name__ == "__main__":
    main()
