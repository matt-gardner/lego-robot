"""Opt-in recognition-only SFT; no RL, action updates, GPU provisioning or rendering."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from so101.cloud_checkpoint import atomic_json
from so101.control_model.piece_recognition import (
    PieceRecognitionModel, prepare_image, recognition_loss,
)
from so101.piece_recognition_data import digest, read_bound


def image_tensor(record: dict, device: str) -> torch.Tensor:
    path = Path(record["path"])
    read_bound(path, record["sha256"])
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def examples(rows: list[dict], maximum_queries: int) -> list[dict]:
    result = []
    for row in rows:
        for start in range(0, len(row["targets"]), maximum_queries):
            result.append({**row, "targets": row["targets"][start:start + maximum_queries]})
    return result


def make_batch(row: dict, part_indices: dict, category_indices: dict, parts: dict,
               device: str):
    history = torch.stack([image_tensor(record, device) for record in row["history"]])[None]
    height, width = history.shape[-2:]
    boxes = torch.tensor([target["target_bbox_xyxy"] for target in row["targets"]],
                         dtype=torch.float32, device=device)[None]
    boxes = boxes / torch.tensor([width, height, width, height], device=device)
    identities = [target["part_id"] for target in row["targets"]]
    labels = torch.tensor([[part_indices[identity] for identity in identities]], device=device)
    categories = {
        name: torch.tensor([[mapping[parts[identity]["categories"][name]] for identity in identities]],
                           device=device)
        for name, mapping in category_indices.items()
    }
    return history, boxes, torch.ones_like(labels, dtype=torch.bool), labels, categories


@torch.no_grad()
def build_anchors(backbone, canonical: dict, identities: list[str], device: str) -> torch.Tensor:
    backbone.eval()
    anchors = []
    for identity in identities:
        views = []
        for record in canonical[identity]:
            image = prepare_image(image_tensor(record, device)[None])
            views.append(F.normalize(backbone.embed_image(image).float().mean(dim=1), dim=-1)[0])
        anchors.append(torch.stack(views))
    return torch.stack(anchors)


@torch.no_grad()
def evaluate(model, rows, indices, category_indices, parts, device, training_counts):
    model.eval()
    per_part = {identity: Counter() for identity in indices}
    confusion, category_correct, category_count = Counter(), Counter(), Counter()
    slices = {"seen_id": Counter(), "unseen_id": Counter()}
    for row in rows:
        history, boxes, valid, labels, categories = make_batch(
            row, indices, category_indices, parts, device,
        )
        scores, category_scores = model(history, boxes, valid)
        ranking = scores[0].topk(min(5, len(indices)), dim=-1).indices
        identities = list(indices)
        for query, target in enumerate(row["targets"]):
            identity = target["part_id"]
            truth = labels[0, query].item()
            measured = {"n": 1, **{
                f"top{k}": int(truth in ranking[query, :min(k, len(indices))].tolist())
                for k in (1, 3, 5)
            }}
            per_part[identity].update(measured)
            slices["seen_id" if training_counts[identity] else "unseen_id"].update(measured)
            prediction = identities[ranking[query, 0].item()]
            if prediction != identity:
                confusion[(identity, prediction)] += 1
        for name, logits in category_scores.items():
            category_correct[name] += int((logits.argmax(-1)[valid] == categories[name][valid]).sum())
            category_count[name] += int(valid.sum())

    def summarize(counts):
        return {"n": counts["n"], **{
            f"top{k}": counts[f"top{k}"] / counts["n"] if counts["n"] else None
            for k in (1, 3, 5)
        }}

    total = sum(per_part.values(), Counter())
    observed = [counts for counts in per_part.values() if counts["n"]]
    prior = max(training_counts, key=training_counts.get)
    return {
        **summarize(total),
        "families": len({row["family"] for row in rows}),
        "macro_top1": sum(c["top1"] / c["n"] for c in observed) / len(observed),
        "per_part": {identity: {**summarize(counts), "train_queries": training_counts[identity]}
                     for identity, counts in per_part.items() if counts["n"]},
        "seen_unseen": {name: summarize(counts) for name, counts in slices.items()},
        "category_accuracy": {name: {"n": category_count[name],
                                     "accuracy": count / category_count[name]}
                              for name, count in category_correct.items()},
        "confusions": [{"truth": truth, "prediction": prediction, "count": count}
                       for (truth, prediction), count in confusion.most_common(30)],
        "training_majority_baseline": {
            "part_id": prior, "n": total["n"], "top1": per_part[prior]["n"] / total["n"],
        },
        "scope": ("n counts visible box-view observations, not independent pieces. "
                  "Raw synthetic recognition; no calibrated acceptance, real transfer or action evaluation."),
    }


def train(args):
    # Checkpoint dependencies are optional until an actual training run is requested.
    from so101.control_model.modeling_entity_smolvla import load_entity_checkpoint

    if (args.steps < 1 or args.accumulation < 1 or args.maximum_queries < 1
            or args.validation_interval < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0):
        raise ValueError("Training dimensions and learning rate must be positive")
    if args.output.exists():
        raise FileExistsError("Use a fresh recognition output; this entry point never resumes a trainer")
    data = json.loads(args.images.read_bytes())
    if data.get("kind") != "pile-recognition-images":
        raise ValueError("Expected recognition image manifest")
    read_bound(Path(data["plan"]), data["plan_sha256"])
    family_splits, image_splits = {}, {}
    for row in data["rows"]:
        if row["split"] not in ("train", "development", "test"):
            raise ValueError("Unknown recognition split")
        previous = family_splits.setdefault(row["family"], row["split"])
        if previous != row["split"]:
            raise ValueError("Pile family crosses recognition splits")
        for image in row["history"]:
            previous = image_splits.setdefault(image["sha256"], row["split"])
            if previous != row["split"]:
                raise ValueError("Identical image bytes cross recognition splits")
    splits = {name: examples([row for row in data["rows"] if row["split"] == name],
                             args.maximum_queries)
              for name in ("train", "development", "test")}
    if not all(splits.values()):
        raise ValueError("Recognition requires nonempty train/development/test splits")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sampling = random.Random(args.seed)
    indices = {identity: index for index, identity in enumerate(sorted(data["parts"]))}
    category_indices = {
        name: {label: index for index, label in enumerate(sorted({
            part["categories"][name] for part in data["parts"].values()
        }))}
        for name in next(iter(data["parts"].values()))["categories"]
    }
    counts = Counter(target["part_id"] for row in splits["train"] for target in row["targets"])
    source_files = {name: digest(args.checkpoint / name)
                    for name in ("model.pt", "entity_config.json", "metadata.json")}
    policy, entity_config, _normalization = load_entity_checkpoint(args.checkpoint, args.device)
    if entity_config.history_steps != 4:
        raise ValueError("Recognition experiment requires the existing four-frame VLA")
    anchors = build_anchors(policy.model.vlm_with_expert, data["canonical"], list(indices), args.device)
    model = PieceRecognitionModel(
        policy.model.vlm_with_expert, anchors,
        {name: len(mapping) for name, mapping in category_indices.items()},
        train_backbone=not args.freeze_backbone,
    ).to(args.device)
    # Keep only the recognition wrapper; no action generation or optimizer state is reused.
    del policy
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=.01,
    )
    args.output.mkdir(parents=True)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "objective": "recognition-embedding-plus-category-SFT; no actions or RL",
        "batch_size": args.accumulation,
        "holdout_scope": "Held out from this recognition SFT; prior checkpoint scene exposure is unverified.",
        "images_sha256": digest(args.images), "checkpoint_files_sha256": source_files,
        "parts": list(indices), "category_indices": category_indices,
        "anchor_encoder": "frozen initial-checkpoint vision+connector; mean patches, 12 separate views",
        "torch_version": str(torch.__version__),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad),
        "implementation_sha256": {
            "train_piece_recognition.py": digest(Path(__file__)),
            "control_model/piece_recognition.py": digest(
                Path(__file__).with_name("control_model") / "piece_recognition.py"),
        },
    }
    torch.save({"anchors": anchors.cpu(), "config": config}, args.output / "anchors.pt")
    log = {"kind": "piece_recognition_sft", "run_config": config,
           "steps": [], "validation": [], "status": "training"}
    atomic_json(args.output / "sft_training.json", log)
    best = -1.
    for step in range(1, args.steps + 1):
        if (args.output / "STOP").exists():
            log["status"] = "stopped"
            atomic_json(args.output / "sft_training.json", log)
            return
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for _ in range(args.accumulation):
            batch = make_batch(sampling.choice(splits["train"]), indices, category_indices,
                               data["parts"], args.device)
            history, boxes, valid, targets, categories = batch
            scores, predictions = model(history, boxes, valid)
            loss = recognition_loss(scores, predictions, targets, categories, valid)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite recognition loss")
            (loss / args.accumulation).backward()
            total_loss += loss.item() / args.accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        log["steps"].append({"step": step, "loss": total_loss,
                             "samples_seen": step * args.accumulation,
                             "learning_rate": args.learning_rate,
                             "seconds": time.monotonic() - started})
        if step % args.validation_interval == 0 or step == args.steps:
            log["status"] = "evaluating"
            atomic_json(args.output / "sft_training.json", log)
            metrics = evaluate(model, splits["development"], indices, category_indices,
                               data["parts"], args.device, counts)
            log["validation"].append({"step": step, **metrics})
            if metrics["macro_top1"] > best:
                best = metrics["macro_top1"]
                temporary = args.output / "best.pt.tmp"
                torch.save({"model": model.state_dict(), "config": config, "step": step}, temporary)
                temporary.replace(args.output / "best.pt")
        log["status"] = "training"
        atomic_json(args.output / "sft_training.json", log)
    saved = torch.load(args.output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(saved["model"])
    test = evaluate(model, splits["test"], indices, category_indices,
                    data["parts"], args.device, counts)
    atomic_json(args.output / "test_results.json", {
        **test, "selected_step": saved["step"], "selection": "development macro_top1",
        "images_sha256": config["images_sha256"], "checkpoint_files_sha256": source_files,
    })
    log["status"] = "completed"
    atomic_json(args.output / "sft_training.json", log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--maximum-queries", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--freeze-backbone", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
