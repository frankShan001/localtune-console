#!/usr/bin/env python
"""Run one-shot inference with a local base model plus a PEFT LoRA adapter."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging():
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    return log_file


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_prompt(tokenizer, prompt, system_prompt="", enable_thinking=False):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            pass
    text = ""
    for message in messages:
        text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def strip_thinking(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return text.split("<think>", 1)[0].strip()
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Infer with base model + PEFT adapter")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--adapter", help="PEFT adapter path")
    parser.add_argument("--prompt", help="Prompt text")
    parser.add_argument("--prompts-file", help="JSON file containing prompt objects")
    parser.add_argument("--compare", action="store_true", help="Generate with base model and adapter for comparison")
    parser.add_argument("--base-only", action="store_true", help="Generate with the base model only")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system-prompt", default="", help="Optional system prompt")
    parser.add_argument("--stop-words", default="[]", help="JSON array of stop strings")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", help="Allow Qwen thinking output")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repositories to execute custom Python code",
    )
    args = parser.parse_args()

    log_file = setup_logging()
    started = time.time()

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_model = resolve_path(args.base_model)
        adapter = resolve_path(args.adapter) if args.adapter else None
        if not base_model.exists():
            raise FileNotFoundError(f"Base model path does not exist: {base_model}")
        if not args.base_only and (not adapter or not (adapter / "adapter_model.safetensors").exists()):
            raise FileNotFoundError(f"Adapter file does not exist: {adapter or '-'}")
        if not args.prompt and not args.prompts_file:
            raise ValueError("Either --prompt or --prompts-file is required")

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map[args.dtype]
        torch.manual_seed(args.seed)
        stop_words = json.loads(args.stop_words)
        if not isinstance(stop_words, list):
            raise ValueError("--stop-words must be a JSON array")

        logging.info("Loading tokenizer: %s", base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model),
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logging.info("Loading 4-bit base model: %s", base_model)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )

        model.eval()

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        def generate(active_model, prompt):
            prompt_text = build_prompt(tokenizer, prompt, system_prompt=args.system_prompt, enable_thinking=args.enable_thinking)
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(active_model.device)
            with torch.no_grad():
                outputs = active_model.generate(**inputs, **gen_kwargs)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            response = raw if args.enable_thinking else strip_thinking(raw)
            stop_positions = [response.find(word) for word in stop_words if word and response.find(word) >= 0]
            if stop_positions:
                response = response[:min(stop_positions)].rstrip()
            return response, raw

        if args.prompts_file:
            prompts = json.loads(resolve_path(args.prompts_file).read_text(encoding="utf-8"))
        else:
            prompts = [{"id": "prompt-1", "prompt": args.prompt}]
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("Prompts file must contain a non-empty JSON array")

        base_outputs = {}
        if args.compare or args.base_only:
            logging.info("Generating base model outputs: %s prompts", len(prompts))
            for index, item in enumerate(prompts):
                prompt = str(item.get("prompt") or "")
                base_outputs[index] = generate(model, prompt)

        adapter_outputs = {}
        if not args.base_only:
            logging.info("Loading adapter: %s", adapter)
            model = PeftModel.from_pretrained(model, str(adapter))
            model.eval()
            logging.info("Generating adapter outputs: %s prompts", len(prompts))
            for index, item in enumerate(prompts):
                prompt = str(item.get("prompt") or "")
                adapter_outputs[index] = generate(model, prompt)

        elapsed = time.time() - started
        results = []
        for index, item in enumerate(prompts):
            entry = {
                "id": item.get("id") or f"prompt-{index + 1}",
                "prompt": item.get("prompt") or "",
                "expected": item.get("expected") or "",
            }
            if index in base_outputs:
                entry["base_response"], entry["base_raw_response"] = base_outputs[index]
            if index in adapter_outputs:
                entry["adapter_response"], entry["adapter_raw_response"] = adapter_outputs[index]
            if not args.compare:
                entry["response"] = entry.get("base_response") if args.base_only else entry.get("adapter_response")
            results.append(entry)

        payload = {
            "ok": True,
            "elapsed_seconds": elapsed,
            "base_model": str(base_model),
            "adapter": str(adapter) if adapter else "",
            "log_file": str(log_file),
            "compare": bool(args.compare),
            "results": results,
        }
        if len(results) == 1:
            payload.update(results[0])
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        logging.exception("Adapter inference failed")
        print(json.dumps({
            "ok": False,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
            "log_file": str(log_file),
        }, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
