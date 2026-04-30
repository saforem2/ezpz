"""Distributed inference over a HuggingFace model + dataset.

A general-purpose inference example: shard a HuggingFace dataset across
ranks (data parallel), generate completions with a HuggingFace model on
each rank, and aggregate the results.

Launch it with::

    ezpz launch python3 -m ezpz.examples.inference \\
        --model meta-llama/Llama-3.2-1B \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1 \\
        --max-samples 256 --max-new-tokens 32

Each rank loads the model, processes its shard of prompts, and
records per-batch latency, throughput (tokens/sec), MFU, and the
generated text.  Outputs go to ``outputs/ezpz.examples.inference/<ts>/``:
``predictions-rank<N>.jsonl`` (one row per sample) and a finalized
``History`` dataset with the timing/perf metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

import ezpz
from ezpz.examples import get_example_outdir
from ezpz.flops import compute_mfu, try_estimate

logger = ezpz.get_logger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse inference command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ezpz.examples.inference",
        description=(
            "Distributed inference over a HuggingFace model + dataset. "
            "Three modes: --mode benchmark (throughput), generate "
            "(synthetic data corpus), eval (accuracy on labeled data). "
            "Each rank processes a disjoint shard of prompts."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["benchmark", "generate", "eval"],
        help=(
            "Inference mode: "
            "'benchmark' = synthetic random tokens, no dataset, focus "
            "on tokens/sec/MFU. "
            "'generate' = dataset prompts → completions, save to JSONL "
            "(synthetic data / distillation use case). "
            "'eval' = dataset prompts + gold labels, compare generated "
            "text to label, report accuracy."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name or local path (ignored in --mode benchmark)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration (subset name)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split (train/validation/test)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Dataset column containing the prompt text",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Dataset column containing the gold label (required for --mode eval)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
        help="Number of benchmark iterations (only with --mode benchmark)",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=3,
        help="Warmup iterations to skip when reporting (only with --mode benchmark)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum number of samples to process across all ranks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-rank batch size",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=512,
        help="Truncate prompts to this many tokens",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --do-sample)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (only used with --do-sample)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=True,
        help="Write per-sample predictions to JSONL",
    )
    parser.add_argument(
        "--no-save-predictions",
        dest="save_predictions",
        action="store_false",
        help="Skip writing per-sample predictions",
    )
    return parser.parse_args(argv)


def shard_indices(total: int, rank: int, world_size: int) -> list[int]:
    """Return the subset of indices [0, total) assigned to *rank*.

    Uses contiguous block sharding — rank ``r`` gets indices
    ``[r*chunk, (r+1)*chunk)`` where ``chunk = ceil(total / world_size)``.
    The last rank may receive fewer samples.
    """
    if total <= 0 or world_size <= 0:
        return []
    chunk = (total + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, total)
    return list(range(start, end))


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for distributed HF inference."""
    # Silence noisy per-request HTTP logs from HF Hub clients
    import logging as _logging
    for _noisy in ("httpx", "huggingface_hub", "filelock", "urllib3"):
        _logging.getLogger(_noisy).setLevel(_logging.WARNING)
    # Silence noisy transformers messages (e.g. the BPE
    # clean_up_tokenization_spaces warning fired on every decode call)
    try:
        import transformers as _transformers
        _transformers.logging.set_verbosity_error()
        _transformers.logging.disable_progress_bar()
    except Exception:
        pass

    args = parse_args(argv)

    # ── Distributed setup ──────────────────────────────────────────
    rank = ezpz.setup_torch(seed=args.seed)
    world_size = ezpz.get_world_size()
    device = ezpz.get_torch_device(as_torch_device=True)
    dtype = _torch_dtype(args.dtype)

    # ── Model + tokenizer ──────────────────────────────────────────
    # Load on rank 0 first to populate the HF cache, then let other
    # ranks read from cache. Avoids thundering-herd downloads on first
    # run from a clean cache.
    if rank == 0:
        logger.info("Loading model: %s", args.model)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _load_tokenizer_and_model():
        tok = AutoTokenizer.from_pretrained(
            args.model,
            clean_up_tokenization_spaces=False,
        )
        # Resolve a usable pad id: prefer existing pad → reuse eos →
        # add a new <pad> token (last-resort, requires resizing the
        # model embedding table to match — done below after model load).
        added_pad_token = False
        if tok.pad_token_id is None:
            if tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "<pad>"})
                added_pad_token = True
        # Left-pad so generation continues from the end of each prompt
        tok.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
        ).to(device)
        # If we extended the vocab, resize the embedding table so the
        # new pad id doesn't index out of bounds.
        if added_pad_token:
            m.resize_token_embeddings(len(tok))
        m.eval()
        return tok, m

    if rank == 0:
        tokenizer, model = _load_tokenizer_and_model()
        if world_size > 1:
            ezpz.synchronize()
    else:
        if world_size > 1:
            ezpz.synchronize()
        tokenizer, model = _load_tokenizer_and_model()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model loaded: %.2fB parameters", n_params / 1e9)

    # Estimate model FLOPS once for MFU/throughput tracking.
    #
    # NOTE: this is a coarse approximation — try_estimate reports
    # fwd+bwd FLOPS for a single forward pass on max_input_tokens
    # context. Inference is forward-only (~1/3 of fwd+bwd), but
    # autoregressive generation has growing context: token i sees
    # context_len + i tokens. We treat each generated token as one
    # forward pass at max_input_tokens length, which:
    #   - over-estimates the early tokens (smaller context with KV cache)
    #   - under-estimates the late tokens (longer context)
    # Reasonable for an example; not a substitute for real profiling.
    _model_flops_fwd_bwd = try_estimate(
        model, (args.batch_size, args.max_input_tokens),
    )
    _model_flops_fwd = _model_flops_fwd_bwd / 3 if _model_flops_fwd_bwd > 0 else 0

    # ── Prompts (and optional labels) ──────────────────────────────
    # In benchmark mode we synthesize random tokens — no dataset load.
    # In generate / eval mode we pull from a HuggingFace dataset.
    # eval_unlabeled = True means we'll score next-token prediction
    # directly (no labels, no generation).
    prompts: list[str]
    labels: list[Optional[str]]
    eval_unlabeled = args.mode == "eval" and not args.label_column

    if args.mode == "benchmark":
        if rank == 0:
            logger.info(
                "Benchmark mode: %d iterations × batch_size=%d at "
                "max_input_tokens=%d, max_new_tokens=%d",
                args.benchmark_iters, args.batch_size,
                args.max_input_tokens, args.max_new_tokens,
            )
        # Synthesize random token IDs as "prompts" — we'll skip the
        # tokenizer in the loop and feed input_ids directly.
        prompts = [""] * args.benchmark_iters * args.batch_size
        labels = [None] * len(prompts)
    else:
        if rank == 0 and eval_unlabeled:
            logger.info(
                "--mode eval without --label-column: scoring next-token "
                "prediction (accuracy + perplexity) on the prompt itself"
            )

        if rank == 0:
            logger.info(
                "Loading dataset: %s [%s] split=%s",
                args.dataset, args.dataset_config, args.dataset_split,
            )
        from datasets import load_dataset

        def _load_ds():
            return load_dataset(
                args.dataset, args.dataset_config, split=args.dataset_split,
            )

        if rank == 0:
            ds = _load_ds()
            if world_size > 1:
                ezpz.synchronize()
        else:
            if world_size > 1:
                ezpz.synchronize()
            ds = _load_ds()

        prompts = []
        labels = []
        for row in ds:
            if not isinstance(row, dict):
                continue
            text = row.get(args.text_column, "")
            if not (isinstance(text, str) and text.strip()):
                continue
            prompts.append(text.strip())
            if args.label_column:
                lbl = row.get(args.label_column)
                labels.append(str(lbl) if lbl is not None else None)
            else:
                labels.append(None)
            if len(prompts) >= args.max_samples:
                break

        if not prompts:
            if rank == 0:
                logger.error(
                    "No prompts found in column %r of %s/%s split=%s — aborting.",
                    args.text_column, args.dataset, args.dataset_config,
                    args.dataset_split,
                )
            ezpz.cleanup()
            return 1

    my_indices = shard_indices(len(prompts), rank, world_size)
    my_prompts = [prompts[i] for i in my_indices]
    my_labels = [labels[i] for i in my_indices]

    if rank == 0:
        logger.info(
            "Total samples: %d → %d per rank (rank 0 has %d)",
            len(prompts),
            (len(prompts) + world_size - 1) // world_size,
            len(my_prompts),
        )

    # ── Output paths ───────────────────────────────────────────────
    module_name = "ezpz.examples.inference"
    outdir = get_example_outdir(module_name)
    if rank == 0:
        logger.info("Outputs will be saved to %s", outdir)

    history = ezpz.History(
        project_name=module_name,
        config={
            "model": args.model,
            "dataset": f"{args.dataset}/{args.dataset_config}",
            "world_size": world_size,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
        },
        outdir=outdir,
        report_dir=outdir,
        report_enabled=True,
        distributed_history=(1 < world_size <= 384),
    )

    # Predictions file: only useful for generate / eval.  Disable in
    # benchmark mode (the "completions" are random-token noise).
    pred_file = None
    if args.save_predictions and args.mode != "benchmark":
        pred_path = Path(outdir) / f"predictions-rank{rank}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_file = pred_path.open("w", encoding="utf-8")

    # ── Inference loop ─────────────────────────────────────────────
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": pad_id,
    }
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    total_samples = 0
    total_new_tokens = 0
    n_correct = 0  # only meaningful in --mode eval (labeled)
    n_with_label = 0
    # For unlabeled --mode eval (next-token prediction):
    n_token_correct = 0
    n_tokens_scored = 0
    nll_sum = 0.0  # sum of -log p(next token); perplexity = exp(nll/n)
    benchmark_warmup = (
        args.benchmark_warmup if args.mode == "benchmark" else 0
    )
    t_start = time.perf_counter()

    with torch.inference_mode():
        for batch_idx in range(0, len(my_prompts), args.batch_size):
            batch_prompts = my_prompts[batch_idx : batch_idx + args.batch_size]
            batch_labels = my_labels[batch_idx : batch_idx + args.batch_size]
            if not batch_prompts:
                continue

            ezpz.synchronize()
            t0 = time.perf_counter()

            if args.mode == "benchmark":
                # Synthesize random input_ids — skip tokenizer entirely
                input_ids = torch.randint(
                    0,
                    int(getattr(tokenizer, "vocab_size", 32000)),
                    (len(batch_prompts), args.max_input_tokens),
                    device=device,
                    dtype=torch.long,
                )
                attention_mask = torch.ones_like(input_ids)
            else:
                enc = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens,
                    return_tensors="pt",
                ).to(device)
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask

            # Unlabeled eval path: forward pass + score next-token
            # prediction. No autoregressive generation.
            decoded: list[str] = []
            batch_token_correct = 0
            batch_token_scored = 0
            batch_nll_sum = 0.0
            n_new = 0
            if eval_unlabeled:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # [B, T, V]
                # Predict token i+1 from logits at position i
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = input_ids[:, 1:].contiguous()
                # Mask padded positions out of the score
                shift_mask = attention_mask[:, 1:].to(torch.bool)
                preds = shift_logits.argmax(dim=-1)
                correct = (preds == shift_targets) & shift_mask
                batch_token_correct = int(correct.sum().item())
                batch_token_scored = int(shift_mask.sum().item())
                # Cross-entropy in float32 for numerical stability
                ce = torch.nn.functional.cross_entropy(
                    shift_logits.float().reshape(-1, shift_logits.size(-1)),
                    shift_targets.reshape(-1),
                    reduction="none",
                ).reshape(shift_targets.shape)
                batch_nll_sum = float((ce * shift_mask).sum().item())
                ezpz.synchronize()
                dt = time.perf_counter() - t0
            else:
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
                ezpz.synchronize()
                dt = time.perf_counter() - t0
                # Slice off the prompt portion so we report only new tokens
                input_len = input_ids.shape[1]
                new_token_ids = output_ids[:, input_len:]
                n_new = int((new_token_ids != pad_id).sum().item())
                # Decode (skip in benchmark mode — random tokens aren't useful)
                if args.mode != "benchmark":
                    decoded = tokenizer.batch_decode(
                        new_token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

            n_in = int(attention_mask.sum().item())
            batch_num = batch_idx // args.batch_size
            is_warmup = batch_num < benchmark_warmup

            # Labeled-eval string match (only when we have labels)
            batch_correct = 0
            batch_with_label = 0
            if args.mode == "eval" and not eval_unlabeled:
                for completion, label in zip(decoded, batch_labels):
                    if label is None:
                        continue
                    batch_with_label += 1
                    if _normalize(completion) == _normalize(label) or (
                        _normalize(label) and _normalize(label) in _normalize(completion)
                    ):
                        batch_correct += 1
                n_correct += batch_correct
                n_with_label += batch_with_label

            if eval_unlabeled:
                n_token_correct += batch_token_correct
                n_tokens_scored += batch_token_scored
                nll_sum += batch_nll_sum

            if not is_warmup:
                total_samples += len(batch_prompts)
                total_new_tokens += n_new

            # In unlabeled-eval mode there are no generated tokens —
            # we ran a single forward pass and scored prompt tokens.
            # Report tokens/sec and MFU based on tokens *scored*
            # (one forward pass per token), not generated.
            if eval_unlabeled:
                tokens_for_throughput = batch_token_scored
            else:
                tokens_for_throughput = n_new

            metrics: dict[str, Any] = {
                "batch": batch_num,
                "samples": len(batch_prompts),
                "input_tokens": n_in,
                "new_tokens": n_new,
                "dt": dt,
                "tokens_per_sec": (
                    tokens_for_throughput / dt if dt > 0 else 0.0
                ),
                "warmup": is_warmup,
            }
            if eval_unlabeled:
                metrics["tokens_scored"] = batch_token_scored
            if _model_flops_fwd > 0 and dt > 0:
                # FLOPS accounting differs between modes:
                # - eval_unlabeled: ONE forward pass over the full batch
                #   (try_estimate was called with (batch_size, max_input_tokens),
                #   so _model_flops_fwd already represents that single pass)
                # - generate / benchmark: ONE forward pass per generated
                #   token (autoregressive), so multiply by n_new
                if eval_unlabeled:
                    step_flops = _model_flops_fwd
                else:
                    step_flops = _model_flops_fwd * max(n_new, 1)
                metrics["tflops"] = step_flops / dt / 1e12
                metrics["mfu"] = compute_mfu(step_flops, dt)
            if args.mode == "eval" and batch_with_label > 0:
                metrics["accuracy"] = batch_correct / batch_with_label
            if eval_unlabeled and batch_token_scored > 0:
                metrics["next_token_accuracy"] = (
                    batch_token_correct / batch_token_scored
                )
                metrics["perplexity"] = float(
                    torch.exp(
                        torch.tensor(batch_nll_sum / batch_token_scored)
                    ).item()
                )

            if not is_warmup:
                history.update(metrics)

            if pred_file is not None:
                for i, (prompt, completion) in enumerate(zip(batch_prompts, decoded)):
                    row = {
                        "rank": rank,
                        "prompt": prompt,
                        "completion": completion,
                    }
                    if args.mode == "eval":
                        row["label"] = batch_labels[i]
                    pred_file.write(json.dumps(row) + "\n")

            if rank == 0:
                tag = " [warmup]" if is_warmup else ""
                parts = [
                    f"batch={batch_num}{tag}",
                    f"samples={metrics['samples']}",
                ]
                # In unlabeled-eval mode there are no generated
                # tokens — show what was *scored* instead.
                if "tokens_scored" in metrics:
                    parts.append(f"tokens_scored={metrics['tokens_scored']}")
                else:
                    parts.append(f"new_tokens={metrics['new_tokens']}")
                parts.extend([
                    f"dt={dt:.3f}s",
                    f"tps={metrics['tokens_per_sec']:.1f}",
                ])
                if "tflops" in metrics:
                    parts.append(f"tflops={metrics['tflops']:.2f}")
                if "mfu" in metrics:
                    parts.append(f"mfu={metrics['mfu']:.2f}%")
                if "accuracy" in metrics:
                    parts.append(f"accuracy={metrics['accuracy']:.3f}")
                if "next_token_accuracy" in metrics:
                    parts.append(
                        f"next_token_acc={metrics['next_token_accuracy']:.3f}"
                    )
                if "perplexity" in metrics:
                    parts.append(f"perplexity={metrics['perplexity']:.2f}")
                logger.info(" ".join(parts))

    if pred_file is not None:
        pred_file.close()

    t_total = time.perf_counter() - t_start
    if rank == 0:
        logger.info(
            "Done [%s] — %d samples, %d new tokens in %.1fs (%.1f tok/s aggregate per rank)",
            args.mode,
            total_samples,
            total_new_tokens,
            t_total,
            total_new_tokens / t_total if t_total > 0 else 0.0,
        )
        if args.mode == "eval" and not eval_unlabeled and n_with_label > 0:
            logger.info(
                "Accuracy [rank 0]: %d/%d = %.2f%%",
                n_correct, n_with_label,
                100.0 * n_correct / n_with_label,
            )
        if eval_unlabeled and n_tokens_scored > 0:
            avg_nll = nll_sum / n_tokens_scored
            ppl = float(torch.exp(torch.tensor(avg_nll)).item())
            logger.info(
                "Next-token [rank 0]: accuracy=%d/%d=%.2f%%  perplexity=%.3f  (NLL/token=%.4f)",
                n_token_correct, n_tokens_scored,
                100.0 * n_token_correct / n_tokens_scored,
                ppl, avg_nll,
            )

    if rank == 0:
        history.finalize(
            outdir=outdir,
            run_name=module_name,
            dataset_fname="inference",
            verbose=False,
        )

    ezpz.cleanup()
    return 0


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace + strip — used for eval matching."""
    if text is None:
        return ""
    return " ".join(str(text).lower().split())


if __name__ == "__main__":
    raise SystemExit(main())
