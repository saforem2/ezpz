"""The default benchmark models must be fetchable without gated HF access."""

from ezpz.examples.run_all import EXAMPLES


def test_hf_benchmark_defaults_use_public_gpt2():
    for name in ("hf", "hf_trainer"):
        example = next(item for item in EXAMPLES if item["name"] == name)
        args = example["args"]
        model_index = args.index("--model_name_or_path")
        assert args[model_index + 1] == "openai-community/gpt2"
        assert "--block_size=512" in args


def test_fsdp_tp_tokenizer_default_is_public_gpt2():
    import ast
    from pathlib import Path

    source = Path("src/ezpz/examples/fsdp_tp.py").read_text()
    tree = ast.parse(source)
    defaults = [
        keyword.value.value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "--tokenizer_name"
        for keyword in call.keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
    ]
    assert defaults == ["openai-community/gpt2"]
