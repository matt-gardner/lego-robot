"""Compare matched microbatch continuations and the authentic existing baseline."""
import argparse
import json
import math
from pathlib import Path

import torch


def parameter_difference(left, right):
    if left.keys() != right.keys() or not left:
        raise ValueError("Compared parameter populations differ or are empty")
    squared_delta = 0.
    squared_reference = 0.
    maximum = 0.
    for name, value in left.items():
        other = right[name]
        if value.shape != other.shape or not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError(f"Invalid compared parameter: {name}")
        reference = value.double()
        delta = other.double() - reference
        squared_delta += float(delta.square().sum())
        squared_reference += float(reference.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    if squared_reference == 0:
        raise ValueError("Reference parameter norm is zero")
    return {
        "tensor_count": len(left), "maximum_absolute_difference": maximum,
        "relative_l2_difference": math.sqrt(squared_delta / squared_reference),
    }


def metrics(log):
    rows = [row for row in log["steps"] if 48 < row["step"] <= 96]
    final = log["validation"][-1]
    if len(rows) != 48 or final["step"] != 96 or len(final["rollouts"]) != 40:
        raise ValueError("Comparison requires 48 updates and the aligned Frozen40 evaluation")
    return {
        "updates": 48, "samples": 6144,
        "mean_step_seconds": sum(row["step_seconds"] for row in rows) / 48,
        "final_training_loss": rows[-1]["loss"],
        "held_out_loss": final["supervised_loss"],
        "successes": sum(bool(row["success"]) for row in final["rollouts"]),
        "evaluation_problems": 40, "evaluation_seconds": final["evaluation_seconds"],
    }


def short_comparison(baseline, tuned, receipts):
    if (len(receipts) != 2 or any(item["status"] != "completed" for item in receipts)
            or receipts[0]["base_sha256"] != receipts[1]["base_sha256"]
            or receipts[0]["sample_fingerprints"] != receipts[1]["sample_fingerprints"]
            or len(receipts[0]["sample_fingerprints"]) != 6):
        raise ValueError("Short comparison requires six completed, sample-matched updates")
    for state in (baseline, tuned):
        if state["step"] != 54 or state["log"]["run_config"]["batch_size"] != 128:
            raise ValueError("Short comparison requires effective batch 128 at update 54")
    a, b = baseline["log"], tuned["log"]
    if a["dataset"] != b["dataset"] or a["steps"][:48] != b["steps"][:48]:
        raise ValueError("Starting checkpoint history or normalized dataset differs")
    rows = [log["steps"][48:] for log in (a, b)]
    for continuation in rows:
        if [row["step"] for row in continuation] != list(range(49, 55)):
            raise ValueError("Short comparison optimizer interval differs")
    fields = ("loss", "action_loss", "box_loss", "depth_loss", "visibility_loss", "gradient_norm")
    for continuation in rows:
        if not all(math.isfinite(row[field]) for row in continuation for field in fields):
            raise ValueError("Short comparison contains non-finite losses or gradients")
    return {
        "status": "completed", "updates": 6, "samples": 768, "measured_updates": 5,
        "checkpoint_step": 48, "final_step": 54,
        "checkpoint_sha256": receipts[0]["base_sha256"],
        "matched_sample_batches": 6,
        "baseline": receipts[0], "accumulated": receipts[1],
        "parameters": parameter_difference(baseline["model"], tuned["model"]),
        "parameter_elements": sum(value.numel() for value in baseline["model"].values()),
        "maximum_absolute_differences": {
            field: max(abs(x[field] - y[field]) for x, y in zip(*rows, strict=True))
            for field in fields
        },
        "first_update": [{field: continuation[0][field] for field in fields}
                         for continuation in rows],
        "notes": [
            "Six matched optimizer updates from the same full model/Adam/RNG checkpoint.",
            "Five post-warmup updates measure full optimizer-update throughput.",
            "This short check does not measure held-out loss or Frozen40 success changes.",
        ],
    }


def write_result(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "tuned", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    baseline = torch.load(args.baseline / "training_state.pt", map_location="cpu", weights_only=False)
    tuned = torch.load(args.tuned / "training_state.pt", map_location="cpu", weights_only=False)
    receipts = [json.loads((path.parent / "benchmark.json").read_text())
                for path in (args.baseline, args.tuned)]
    if args.short:
        write_result(args.output, short_comparison(baseline, tuned, receipts))
        return
    reference = torch.load(args.reference / "checkpoints/training_state_000096.pt",
                           map_location="cpu", weights_only=False)
    if any(state["step"] != 96 for state in (baseline, tuned, reference)):
        raise ValueError("Compared final states are not at optimizer update 96")
    for state in (baseline, tuned):
        if state["log"]["steps"][:48] != reference["log"]["steps"][:48]:
            raise ValueError("Continuation changed the authentic checkpoint history")
        if state["log"]["dataset"] != reference["log"]["dataset"]:
            raise ValueError("Continuation dataset or normalization changed")
    if (receipts[0]["base_sha256"] != receipts[1]["base_sha256"]
            or receipts[0]["sample_fingerprints"] != receipts[1]["sample_fingerprints"]
            or len(receipts[0]["sample_fingerprints"]) != 48):
        raise ValueError("Starting checkpoint or sampled examples differ")
    a, b, existing = (state["log"] for state in (baseline, tuned, reference))
    if not all(log["run_config"]["batch_size"] == 128 for log in (a, b, existing)):
        raise ValueError("Effective batch size changed")
    result = {
        "status": "completed", "baseline_microbatch": 16,
        "tuned_microbatch": receipts[1]["microbatch"], "effective_batch": 128,
        "checkpoint_step": 48, "final_step": 96, "matched_sample_batches": 48,
        "checkpoint_sha256": receipts[0]["base_sha256"],
        "baseline_replay": metrics(a), "tuned": metrics(b), "existing_run": metrics(existing),
        "baseline_replay_vs_existing_parameters": parameter_difference(reference["model"], baseline["model"]),
        "tuned_vs_baseline_replay_parameters": parameter_difference(baseline["model"], tuned["model"]),
        "maximum_loss_difference_vs_existing": max(abs(x["loss"] - y["loss"])
            for x, y in zip(a["steps"][48:96], existing["steps"][48:96])),
        "maximum_loss_difference_tuned_vs_replay": max(abs(x["loss"] - y["loss"])
            for x, y in zip(a["steps"][48:96], b["steps"][48:96])),
        "maximum_gradient_norm_difference": max(abs(x["gradient_norm"] - y["gradient_norm"])
            for x, y in zip(a["steps"][48:96], b["steps"][48:96])),
        "notes": [
            "Same effective batch, saved model/Adam/RNG checkpoint, sampled examples, schedule and frozen evaluation.",
            "Different microbatch execution can change floating-point order and subsequent CUDA random-number consumption.",
            "This is a 48-update sensitivity check, not proof of identical full-sweep rankings.",
        ],
    }
    write_result(args.output, result)


if __name__ == "__main__":
    main()
