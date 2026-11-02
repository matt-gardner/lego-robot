from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable, Iterator
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks, pad_vector, resize_with_pad
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching

from so101.control_model.flow_rl import (
    FlowTransition,
    gaussian_log_prob,
    sample_flow_path,
    transition_distribution,
)


@dataclass(frozen=True)
class EntityModelConfig:
    max_entities: int = 4
    history_steps: int = 4
    role_token_length: int = 8
    depth_input: bool = True

    def __post_init__(self) -> None:
        if self.max_entities <= 0 or self.history_steps <= 0 or self.role_token_length <= 0:
            raise ValueError("Entity model dimensions must be positive")


class EntityVLAFlowMatching(VLAFlowMatching):
    def init_entity_modules(self, entity_config: EntityModelConfig) -> None:
        self.entity_config = entity_config
        text_hidden = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size
        self.entity_numeric_proj = nn.Sequential(
            nn.Linear(11, text_hidden),
            nn.SiLU(),
            nn.Linear(text_hidden, text_hidden),
        )
        self.entity_norm = nn.LayerNorm(text_hidden)
        self.history_motion_proj = nn.Sequential(
            nn.Linear(12, text_hidden),
            nn.SiLU(),
            nn.Linear(text_hidden, text_hidden),
        )
        self.history_time_embedding = nn.Embedding(entity_config.history_steps, text_hidden)
        self.entity_to_expert = nn.Linear(text_hidden, expert_hidden)
        self.tracking_head = nn.Sequential(
            nn.Linear(expert_hidden * 2, expert_hidden),
            nn.SiLU(),
            nn.Linear(expert_hidden, 9),
        )

    def _entity_embeddings(
        self,
        role_tokens: Tensor,
        role_attention_mask: Tensor,
        boxes: Tensor,
        visibility: Tensor,
        depth_m: Tensor,
        depth_known: Tensor,
        camera_position: Tensor,
        entity_mask: Tensor,
    ) -> Tensor:
        batch, entities, role_length = role_tokens.shape
        flat_tokens = role_tokens.reshape(batch * entities, role_length)
        flat_mask = role_attention_mask.reshape(batch * entities, role_length).bool()
        role_embeddings = self.vlm_with_expert.embed_language_tokens(flat_tokens)
        role_embeddings = role_embeddings * flat_mask.unsqueeze(-1)
        role_embeddings = role_embeddings.sum(dim=1) / flat_mask.sum(dim=1, keepdim=True).clamp_min(1)
        role_embeddings = role_embeddings.reshape(batch, entities, -1).to(
            dtype=self.entity_numeric_proj[0].weight.dtype
        )
        numeric = torch.cat(
            (
                boxes,
                visibility.unsqueeze(-1),
                depth_m.unsqueeze(-1),
                depth_known.unsqueeze(-1),
                camera_position,
            ),
            dim=-1,
        ).to(dtype=self.entity_numeric_proj[0].weight.dtype)
        numeric_embeddings = self.entity_numeric_proj(numeric)
        embeddings = self.entity_norm(role_embeddings + numeric_embeddings)
        return embeddings * entity_mask.unsqueeze(-1).to(dtype=embeddings.dtype)

    def _prefix_with_entities(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor,
        history_state: Tensor,
        history_action: Tensor,
        role_tokens: Tensor,
        role_attention_mask: Tensor,
        boxes: Tensor,
        visibility: Tensor,
        depth_m: Tensor,
        depth_known: Tensor,
        camera_position: Tensor,
        entity_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        prefix, pad_mask, attention_mask = super().embed_prefix(
            images, image_masks, language_tokens, language_masks, state
        )
        entity_embeddings = self._entity_embeddings(
            role_tokens,
            role_attention_mask,
            boxes,
            visibility,
            depth_m,
            depth_known,
            camera_position,
            entity_mask,
        )
        history = torch.cat((history_state, history_action), dim=-1)
        history_embeddings = self.history_motion_proj(
            history.to(dtype=self.history_motion_proj[0].weight.dtype)
        ).to(dtype=prefix.dtype)
        steps = history_embeddings.shape[1]
        time_ids = torch.arange(steps, device=history_embeddings.device)
        history_embeddings = history_embeddings + self.history_time_embedding(time_ids)[None]
        history_mask = torch.ones(
            history_embeddings.shape[:2], dtype=torch.bool, device=history_embeddings.device
        )
        entity_mask = entity_mask.bool()
        prefix = torch.cat((prefix, history_embeddings, entity_embeddings), dim=1)
        pad_mask = torch.cat((pad_mask, history_mask, entity_mask), dim=1)
        added_attention = torch.ones(
            1,
            steps + entity_embeddings.shape[1],
            dtype=torch.bool,
            device=attention_mask.device,
        ).expand(attention_mask.shape[0], -1)
        attention_mask = torch.cat((attention_mask, added_attention), dim=1)
        return prefix, pad_mask, attention_mask, entity_embeddings

    def _tracking_predictions(
        self,
        suffix_out: Tensor,
        entity_embeddings: Tensor,
        *,
        detach_features: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if detach_features:
            suffix_out = suffix_out.detach()
            entity_embeddings = entity_embeddings.detach()
        entity_queries = self.entity_to_expert(
            entity_embeddings.to(dtype=self.entity_to_expert.weight.dtype)
        )
        action_features = suffix_out[:, :, None, :].expand(
            -1, -1, entity_queries.shape[1], -1
        )
        entity_features = entity_queries[:, None, :, :].expand(
            -1, suffix_out.shape[1], -1, -1
        )
        interaction = torch.cat((action_features, entity_features), dim=-1)
        raw = self.tracking_head(interaction)
        center = torch.sigmoid(raw[..., :2]) + raw[..., 7:9]
        mean_extent = torch.sigmoid(raw[..., 2:3])
        anisotropy_raw = raw[..., 3:5]
        anisotropy = (
            mean_extent
            * anisotropy_raw
            / (1 + torch.linalg.vector_norm(anisotropy_raw, dim=-1, keepdim=True))
        )
        boxes = torch.cat((center, mean_extent, anisotropy), dim=-1)
        log_depth = raw[..., 5]
        visibility_logits = raw[..., 6]
        return boxes, log_depth, visibility_logits

    def _denoise_step_entity(
        self,
        prefix_pad_masks: Tensor,
        past_key_values,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        suffix, suffix_pad, suffix_attention = self.embed_suffix(x_t, timestep)
        suffix_length = suffix_pad.shape[1]
        batch = prefix_pad_masks.shape[0]
        prefix_length = prefix_pad_masks.shape[1]
        prefix_attention = prefix_pad_masks[:, None, :].expand(
            batch, suffix_length, prefix_length
        )
        suffix_attention_2d = make_att_2d_masks(suffix_pad, suffix_attention)
        attention = torch.cat((prefix_attention, suffix_attention_2d), dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        positions = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
        outputs, _ = self.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix],
            use_cache=self.config.use_cache,
        )
        if past_key_values is not None:
            past_key_values.crop(prefix_length)
        suffix_out = outputs[1][:, -self.config.chunk_size :].float()
        return self.action_out_proj(suffix_out)

    def _rl_velocity_function(self, inputs: tuple) -> Callable[[Tensor, Tensor], Tensor]:
        # RL trains only the action projections; observation features remain frozen.
        with torch.no_grad():
            prefix, pad, attention, _ = self._prefix_with_entities(*inputs)
            _, cache = self.vlm_with_expert.forward(
                attention_mask=make_att_2d_masks(pad, attention),
                position_ids=torch.cumsum(pad, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[prefix, None],
                use_cache=self.config.use_cache,
            )
        def velocity(latent: Tensor, timestep: Tensor) -> Tensor:
            result = self._denoise_step_entity(pad, cache, latent, timestep)
            if cache is not None:
                # Cropping removes suffix tokens, but not their autograd graph.
                for layer in cache.layers:
                    layer.keys = layer.keys.detach()
                    layer.values = layer.values.detach()
            return result

        return velocity

    @torch.no_grad()
    def sample_actions_entity_sde(
        self, inputs: tuple, noise_scale: float,
        final_action_shape: tuple[int, int] | None = None,
        record_noise: bool = False,
    ) -> tuple[Tensor, list[FlowTransition]]:
        state = inputs[4]
        noise = self.sample_noise(
            (state.shape[0], self.config.chunk_size, self.config.max_action_dim),
            state.device,
        )
        return sample_flow_path(
            self._rl_velocity_function(inputs), noise,
            self.config.num_steps, noise_scale,
            final_action_shape,
            record_noise,
        )

    def denoising_log_probs_entity(
        self, inputs: tuple, path: list[FlowTransition]
    ) -> Iterator[Tensor]:
        velocity = self._rl_velocity_function(inputs)
        device = inputs[4].device
        for transition in path:
            latent = transition.latent.to(device)
            timestep = torch.full(
                (latent.shape[0],), transition.time,
                dtype=torch.float32, device=device,
            )
            mean, std = transition_distribution(
                latent, velocity(latent, timestep), transition.time,
                transition.step_size, transition.noise_scale,
            )
            yield gaussian_log_prob(
                transition.next_latent.to(device), mean, std, transition.active_shape
            )

    def forward_entity(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        history_state: Tensor,
        history_action: Tensor,
        role_tokens: Tensor,
        role_attention_mask: Tensor,
        boxes: Tensor,
        visibility: Tensor,
        depth_m: Tensor,
        depth_known: Tensor,
        camera_position: Tensor,
        entity_mask: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions
        prefix, prefix_pad, prefix_attention, entity_embeddings = self._prefix_with_entities(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            history_state,
            history_action,
            role_tokens,
            role_attention_mask,
            boxes,
            visibility,
            depth_m,
            depth_known,
            camera_position,
            entity_mask,
        )
        suffix, suffix_pad, suffix_attention = self.embed_suffix(noisy_actions, time)
        pad_mask = torch.cat((prefix_pad, suffix_pad), dim=1)
        attention_mask = torch.cat((prefix_attention, suffix_attention), dim=1)
        attention_2d = make_att_2d_masks(pad_mask, attention_mask)
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].float()
        predicted_velocity = self.action_out_proj(suffix_out)
        action_losses = F.mse_loss(target_velocity, predicted_velocity, reduction="none")
        predicted_boxes, predicted_log_depth, visibility_logits = self._tracking_predictions(
            suffix_out, entity_embeddings, detach_features=True
        )
        return (
            action_losses,
            predicted_boxes,
            predicted_log_depth,
            visibility_logits,
            predicted_velocity,
        )

    @torch.no_grad()
    def sample_actions_entity(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor,
        history_state: Tensor,
        history_action: Tensor,
        role_tokens: Tensor,
        role_attention_mask: Tensor,
        boxes: Tensor,
        visibility: Tensor,
        depth_m: Tensor,
        depth_known: Tensor,
        camera_position: Tensor,
        entity_mask: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = state.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (batch, self.config.chunk_size, self.config.max_action_dim), state.device
            )
        prefix, prefix_pad, prefix_attention, entity_embeddings = self._prefix_with_entities(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            history_state,
            history_action,
            role_tokens,
            role_attention_mask,
            boxes,
            visibility,
            depth_m,
            depth_known,
            camera_position,
            entity_mask,
        )
        prefix_attention_2d = make_att_2d_masks(prefix_pad, prefix_attention)
        prefix_positions = torch.cumsum(prefix_pad, dim=1) - 1
        _, cache = self.vlm_with_expert.forward(
            attention_mask=prefix_attention_2d,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=self.config.use_cache,
        )
        actions = euler_integrate(
            lambda value, timestep: self._denoise_step_entity(
                prefix_pad_masks=prefix_pad,
                past_key_values=cache,
                x_t=value,
                timestep=timestep,
            ),
            noise,
            self.config.num_steps,
        )
        timestep = torch.zeros(batch, device=state.device)
        suffix, suffix_pad, suffix_attention = self.embed_suffix(actions, timestep)
        pad_mask = torch.cat((prefix_pad, suffix_pad), dim=1)
        attention_mask = torch.cat((prefix_attention, suffix_attention), dim=1)
        attention_2d = make_att_2d_masks(pad_mask, attention_mask)
        positions = torch.cumsum(pad_mask, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].float()
        predicted_boxes, predicted_log_depth, visibility_logits = self._tracking_predictions(
            suffix_out, entity_embeddings
        )
        return actions, predicted_boxes, predicted_log_depth, visibility_logits


def attach_entity_modules(
    policy: SmolVLAPolicy, entity_config: EntityModelConfig
) -> SmolVLAPolicy:
    if not isinstance(policy.model, EntityVLAFlowMatching):
        policy.model.__class__ = EntityVLAFlowMatching
        policy.model.init_entity_modules(entity_config)
    return policy


def prepare_history_images(policy: SmolVLAPolicy, images: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    if images.ndim != 5:
        raise ValueError("Expected history images with shape (batch, time, channels, height, width)")
    prepared, masks = [], []
    for index in range(images.shape[1]):
        image = images[:, index]
        if policy.config.resize_imgs_with_padding is not None:
            image = resize_with_pad(
                image,
                policy.config.resize_imgs_with_padding[1],
                policy.config.resize_imgs_with_padding[0],
                pad_value=0,
            )
        prepared.append(image * 2 - 1)
        masks.append(torch.ones(image.shape[0], dtype=torch.bool, device=image.device))
    return prepared, masks


def save_entity_checkpoint(
    path: Path,
    policy: SmolVLAPolicy,
    entity_config: EntityModelConfig,
    source_checkpoint: str,
    normalization: dict[str, list[float]],
    *,
    exist_ok: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=exist_ok)
    temporary = path / "model.pt.tmp"
    torch.save(policy.state_dict(), temporary)
    temporary.replace(path / "model.pt")
    (path / "entity_config.json").write_text(json.dumps(asdict(entity_config), indent=2))
    (path / "metadata.json").write_text(json.dumps({
        "source_checkpoint": source_checkpoint,
        "normalization": normalization,
        "action_expert_dtype": str(next(
            policy.model.vlm_with_expert.lm_expert.parameters()
        ).dtype).removeprefix("torch."),
    }, indent=2))


def load_entity_checkpoint(path: Path, device: str) -> tuple[SmolVLAPolicy, EntityModelConfig, dict]:
    metadata = json.loads((path / "metadata.json").read_text())
    entity_config = EntityModelConfig(**json.loads((path / "entity_config.json").read_text()))
    policy = SmolVLAPolicy.from_pretrained(metadata["source_checkpoint"])
    attach_entity_modules(policy, entity_config)
    if "action_expert_dtype" in metadata:
        dtypes = {
            "float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16, "float64": torch.float64,
        }
        name = metadata["action_expert_dtype"]
        if name not in dtypes:
            raise ValueError(f"Unsupported checkpoint action expert dtype: {name}")
        policy.model.vlm_with_expert.lm_expert.to(dtype=dtypes[name])
    source_state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    target_state = policy.state_dict()
    if source_state.keys() == target_state.keys() and all(
        key in source_state and source_state[key].shape == value.shape
        for key, value in target_state.items()
    ):
        policy.load_state_dict(source_state)
    else:
        migrated = {
            key: (
                source_state[key]
                if key in source_state and source_state[key].shape == value.shape
                else value.clone()
            )
            for key, value in target_state.items()
        }
        numeric_key = "model.entity_numeric_proj.0.weight"
        if numeric_key in source_state and source_state[numeric_key].shape[1] == 8:
            migrated[numeric_key][:, :8] = source_state[numeric_key]
            migrated[numeric_key][:, 8:] = 0
        elif numeric_key in source_state and source_state[numeric_key].shape[1] == 7:
            migrated[numeric_key][:, :2] = source_state[numeric_key][:, :2]
            migrated[numeric_key][:, 2] = (
                source_state[numeric_key][:, 2] + source_state[numeric_key][:, 3]
            )
            migrated[numeric_key][:, 3:5] = 0
            migrated[numeric_key][:, 5:8] = source_state[numeric_key][:, 4:7]
            migrated[numeric_key][:, 8:] = 0
        interaction_key = "model.tracking_head.0.weight"
        if (
            interaction_key in source_state
            and source_state[interaction_key].shape[1] * 2
            == migrated[interaction_key].shape[1]
        ):
            hidden = source_state[interaction_key].shape[1]
            migrated[interaction_key][:, :hidden] = source_state[interaction_key]
            migrated[interaction_key][:, hidden:] = source_state[interaction_key]
        output_weight_key = "model.tracking_head.2.weight"
        output_bias_key = "model.tracking_head.2.bias"
        if (
            output_weight_key in source_state
            and source_state[output_weight_key].shape[0] == 6
            and migrated[output_weight_key].shape[0] >= 7
        ):
            migrated[output_weight_key][:2] = source_state[output_weight_key][:2]
            migrated[output_weight_key][2] = source_state[output_weight_key][2:4].mean(dim=0)
            migrated[output_weight_key][3:5] = 0
            migrated[output_weight_key][5:7] = source_state[output_weight_key][4:]
            migrated[output_bias_key][:2] = source_state[output_bias_key][:2]
            migrated[output_bias_key][2] = source_state[output_bias_key][2:4].mean()
            migrated[output_bias_key][3:5] = 0
            migrated[output_bias_key][5:7] = source_state[output_bias_key][4:]
            migrated[output_weight_key][7:] = 0
            migrated[output_bias_key][7:] = 0
        elif (
            output_weight_key in source_state
            and source_state[output_weight_key].shape[0] == 7
            and migrated[output_weight_key].shape[0] == 9
        ):
            migrated[output_weight_key][:7] = source_state[output_weight_key]
            migrated[output_bias_key][:7] = source_state[output_bias_key]
            migrated[output_weight_key][7:] = 0
            migrated[output_bias_key][7:] = 0
        policy.load_state_dict(migrated)
    return policy.to(device).eval(), entity_config, metadata["normalization"]


def pad_actions(actions: Tensor, max_action_dim: int) -> Tensor:
    return pad_vector(actions, max_action_dim)
