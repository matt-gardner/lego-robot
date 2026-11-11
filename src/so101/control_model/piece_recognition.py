"""Box-query recognition using shared SmolVLA features and frozen render anchors."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def prepare_image(image: Tensor, size: int = 512) -> Tensor:
    height, width = image.shape[-2:]
    scale = size / max(height, width)
    resized = F.interpolate(image, size=(max(1, round(height * scale)),
                                        max(1, round(width * scale))),
                            mode="bilinear", align_corners=False)
    vertical, horizontal = size - resized.shape[-2], size - resized.shape[-1]
    return F.pad(resized, (horizontal // 2, horizontal - horizontal // 2,
                           vertical // 2, vertical - vertical // 2)) * 2 - 1


class PieceRecognitionModel(nn.Module):
    def __init__(self, backbone: nn.Module, anchors: Tensor, category_sizes: dict[str, int],
                 *, train_backbone: bool = True):
        super().__init__()
        if anchors.ndim != 3 or not torch.isfinite(anchors).all():
            raise ValueError("Expected finite (parts, views, hidden) render anchors")
        parts, views, hidden = anchors.shape
        if min(parts, views, hidden) < 1:
            raise ValueError("Render anchors cannot be empty")
        self.backbone = backbone
        self.register_buffer("anchors", F.normalize(anchors.float(), dim=-1))
        self.box_projection = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.frame_embedding = nn.Embedding(4, hidden)
        self.query_embedding = nn.Parameter(torch.zeros(hidden))
        self.output_projection = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.category_heads = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, count))
            for name, count in category_sizes.items()
        })
        if not category_sizes or any(count < 1 for count in category_sizes.values()):
            raise ValueError("Category heads must be nonempty")
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.freeze_vision_encoder = not train_backbone
        backbone.train_expert_only = not train_backbone
        if train_backbone:
            for name, parameter in backbone.vlm.named_parameters():
                parameter.requires_grad_("lm_head" not in name)

    def forward(self, history: Tensor, boxes: Tensor, valid: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if history.ndim != 5 or history.shape[1:3] != (4, 3):
            raise ValueError("Expected four RGB history images")
        if (boxes.ndim != 3 or boxes.shape[-1] != 4 or valid.shape != boxes.shape[:2]
                or boxes.shape[0] != history.shape[0] or boxes.shape[1] < 1
                or valid.dtype != torch.bool or not valid.any(dim=1).all()):
            raise ValueError("Expected at least one valid bbox in each batch row")
        if (not torch.isfinite(history).all() or history.min() < 0 or history.max() > 1
                or not torch.isfinite(boxes).all() or boxes.min() < 0 or boxes.max() > 1
                or not (boxes[..., 2:][valid] > boxes[..., :2][valid]).all()):
            raise ValueError("Expected RGB in [0,1] and normalized nonempty xyxy boxes")
        tokens = []
        for index in range(4):
            image = self.backbone.embed_image(prepare_image(history[:, index]))
            frame = self.frame_embedding.weight[index].to(image.dtype)
            tokens.append((image + frame) * math.sqrt(image.shape[-1]))
        prefix = torch.cat(tokens, dim=1)
        queries = (self.box_projection(boxes.float()) + self.query_embedding
                   + self.frame_embedding.weight[3]).to(prefix.dtype)
        inputs = torch.cat((prefix, queries), dim=1)
        batch, image_count = prefix.shape[:2]
        count = boxes.shape[1]
        mask = torch.zeros(batch, image_count + count, image_count + count,
                           dtype=torch.bool, device=inputs.device)
        mask[:, :image_count, :image_count] = True
        mask[:, image_count:, :image_count] = valid[:, :, None]
        # Queries cannot read one another or alter the image representations.
        mask[:, image_count:, image_count:] = torch.eye(
            count, dtype=torch.bool, device=inputs.device)[None]
        # A query's list position must not act as a shortcut for the part identity.
        positions = torch.cat((
            torch.arange(image_count, device=inputs.device),
            torch.full((count,), image_count, device=inputs.device),
        ))[None].expand(batch, -1)
        # SmolVLA's prefix-prefill path uses self-attention in every VLM layer.
        # Without use_cache its alternating cross-attention expects action tokens.
        (output, _), _cache = self.backbone.forward(
            attention_mask=mask, position_ids=positions, past_key_values=None,
            inputs_embeds=[inputs, None], use_cache=True,
        )
        features = output[:, -count:].float()
        embeddings = F.normalize(self.output_projection(features), dim=-1)
        scores = torch.einsum("bqh,pvh->bqpv", embeddings, self.anchors).amax(dim=-1) / .1
        return scores, {name: head(features) for name, head in self.category_heads.items()}


def recognition_loss(scores: Tensor, categories: dict[str, Tensor], part_targets: Tensor,
                     category_targets: dict[str, Tensor], valid: Tensor,
                     category_weight: float = .25) -> Tensor:
    if not valid.any() or not math.isfinite(category_weight) or category_weight < 0:
        raise ValueError("Recognition loss needs valid targets and nonnegative category weight")
    loss = F.cross_entropy(scores[valid], part_targets[valid])
    for name, logits in categories.items():
        selected = valid & (category_targets[name] >= 0)
        if selected.any():
            loss = loss + category_weight * F.cross_entropy(logits[selected], category_targets[name][selected])
    return loss
