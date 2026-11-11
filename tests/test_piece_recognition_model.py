from collections import Counter
from types import SimpleNamespace
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from so101.control_model.piece_recognition import PieceRecognitionModel, recognition_loss
from so101.train_piece_recognition import evaluate, examples


class TinyPrefixBackbone(nn.Module):
    """Small differentiable stand-in for the public SmolVLA prefix interface."""
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(3, 8)
        self.lm_expert = nn.Linear(8, 8)
        self.seen_mask = None

    def embed_image(self, image):
        return self.vlm(image.mean(dim=(-2, -1)))[:, None]

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
        self.seen_mask = attention_mask
        self.seen_positions = position_ids
        assert use_cache and past_key_values is None and inputs_embeds[1] is None
        inputs = inputs_embeds[0]
        weights = attention_mask.float()
        pooled = weights @ inputs / weights.sum(-1, keepdim=True)
        return [inputs + pooled, None], SimpleNamespace()


class RecognitionModelTests(unittest.TestCase):
    def test_per_box_outputs_reach_shared_encoder_not_action_expert_or_fixed_anchors(self):
        torch.manual_seed(7)
        backbone = TinyPrefixBackbone()
        model = PieceRecognitionModel(backbone, torch.randn(3, 2, 8), {"family": 2})
        history = torch.rand(1, 4, 3, 8, 8)
        boxes = torch.tensor([[[.1, .2, .5, .8], [.5, .1, .9, .4]]])
        valid = torch.tensor([[True, True]])
        scores, categories = model(history, boxes, valid)
        self.assertEqual(scores.shape, (1, 2, 3))
        self.assertEqual(categories["family"].shape, (1, 2, 2))
        loss = recognition_loss(scores, categories, torch.tensor([[0, 2]]),
                                {"family": torch.tensor([[1, 0]])}, valid)
        loss.backward()
        self.assertGreater(backbone.vlm.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.box_projection[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(backbone.lm_expert.weight.grad)
        self.assertIsNone(model.anchors.grad)
        self.assertFalse(backbone.seen_mask[0, :4, 4:].any())
        self.assertFalse(backbone.seen_mask[0, 4, 5])
        self.assertTrue(backbone.seen_mask[0, 4, :4].all())
        self.assertEqual(backbone.seen_positions[0, 4:].tolist(), [4, 4])
        swapped, _ = model(history, boxes.flip(1), valid)
        torch.testing.assert_close(swapped, scores.flip(1))

    def test_frozen_probe_only_updates_new_heads_and_masks_padding(self):
        backbone = TinyPrefixBackbone()
        model = PieceRecognitionModel(backbone, torch.randn(3, 2, 8), {"family": 2},
                                      train_backbone=False)
        scores, categories = model(
            torch.rand(1, 4, 3, 8, 8), torch.tensor([[[0., 0., 1., 1.], [0., 0., 0., 0.]]]),
            torch.tensor([[True, False]]),
        )
        loss = recognition_loss(scores, categories, torch.tensor([[1, -999]]),
                                {"family": torch.tensor([[0, -999]])}, torch.tensor([[True, False]]))
        loss.backward()
        self.assertIsNone(backbone.vlm.weight.grad)
        self.assertIsNotNone(model.output_projection[1].weight.grad)

    def test_invalid_box_does_not_produce_a_prediction(self):
        model = PieceRecognitionModel(TinyPrefixBackbone(), torch.randn(2, 2, 8), {"family": 2})
        with self.assertRaisesRegex(ValueError, "nonempty xyxy"):
            model(torch.rand(1, 4, 3, 8, 8), torch.tensor([[[.5, 0., .1, 1.]]]),
                  torch.tensor([[True]]))

    def test_metrics_keep_unseen_ids_and_category_fallback_separate_from_exact_success(self):
        class Predictions(nn.Module):
            def forward(self, history, boxes, valid):
                return (torch.tensor([[[2., 1.], [2., 1.]]]),
                        {"family": torch.tensor([[[1.], [1.]]])})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
            image = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            row = {
                "family": "held-out-pile", "history": [image] * 4,
                "targets": [
                    {"part_id": identity, "target_bbox_xyxy": [0, 0, 8, 8]}
                    for identity in ("common", "rare")
                ],
            }
            metrics = evaluate(Predictions(), [row], {"common": 0, "rare": 1},
                               {"family": {"brick": 0}},
                               {identity: {"categories": {"family": "brick"}}
                                for identity in ("common", "rare")},
                               "cpu", Counter({"common": 5}))
            self.assertEqual(metrics["n"], 2)
            self.assertEqual(metrics["top1"], .5)
            self.assertEqual(metrics["category_accuracy"]["family"]["accuracy"], 1.)
            self.assertEqual(metrics["seen_unseen"]["unseen_id"]["top1"], 0.)
            self.assertEqual(metrics["training_majority_baseline"]["top1"], .5)
            chunks = examples([row], 1)
            self.assertEqual([len(chunk["targets"]) for chunk in chunks], [1, 1])
            self.assertEqual({chunk["family"] for chunk in chunks}, {"held-out-pile"})


if __name__ == "__main__":
    unittest.main()
