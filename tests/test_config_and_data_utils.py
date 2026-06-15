import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_branch
from src.data_utils import format_chatml


def test_model_config_uses_training_max_seq_length():
    raw_config = load_config("configs/model_config.example.yaml")

    config = resolve_branch(raw_config, env_info=None)

    assert raw_config["data"]["max_seq_length"] == 512
    assert raw_config["training"]["max_seq_length"] == 512
    assert config.max_seq_length == 512
    assert config.trust_remote_code is True


def test_remote_model_code_is_disabled_by_default():
    raw_config = load_config("configs/model_config.example.yaml")
    raw_config["model"].pop("trust_remote_code")

    config = resolve_branch(raw_config, env_info=None)

    assert config.trust_remote_code is False


def test_sft_corpus_structures_are_converted_to_chatml():
    instruction = format_chatml({
        "instruction": "Answer briefly.",
        "input": "Why validate data?",
        "output": "To catch errors before training.",
    })["text"]
    rewrite = format_chatml({
        "source": "The room was empty.",
        "target": "Silence occupied the room.",
    })["text"]
    chat = format_chatml({
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    })["text"]

    assert "Answer briefly.\n\nWhy validate data?" in instruction
    assert "To catch errors before training." in instruction
    assert "Rewrite the following text.\n\nThe room was empty." in rewrite
    assert "Silence occupied the room." in rewrite
    assert "<|im_start|>user\nHello<|im_end|>" in chat
