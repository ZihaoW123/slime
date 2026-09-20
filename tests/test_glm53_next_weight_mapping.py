"""Check checkpoint/Transformers layouts without downloading model payloads."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slime_plugins.models.glm5_next.weight_mapping import export_hf_tensor, load_hf_tensor


class DictReader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, name):
        return self.tensors[name]


def test_weight_mapping_round_trip():
    names_and_tensors = {
        "model.language_model.layers.0.attn_hc.base": torch.randn(4),
        "model.language_model.layers.0.self_attn.forget_gate.A_log": torch.randn(2),
        "model.language_model.layers.0.self_attn.conv1d.weight": torch.randn(12, 1, 3),
        "model.language_model.layers.3.mlp.experts.gate_up_proj": torch.randn(2, 8, 4),
        "model.language_model.layers.3.mlp.experts.down_proj": torch.randn(2, 4, 4),
    }
    config = SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=2))
    for hf_name, expected in names_and_tensors.items():
        checkpoint = dict(export_hf_tensor(hf_name, expected))
        actual = load_hf_tensor(hf_name, DictReader(checkpoint), config)
        torch.testing.assert_close(actual, expected)


def test_published_four_layer_index_covers_hf_model():
    transformers = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

    root = Path(__file__).resolve().parents[3] / "GLM-5.3-Flash"
    if not (root / "model.safetensors.index.json").exists():
        pytest.skip("Published GLM-5.3 metadata is not in this workspace")
    config = json.loads((root / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = 4
    text["num_nextn_predict_layers"] = 0
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        text[key] = text[key][:4]
    for key in ("kda_layers", "full_attn_layers"):
        text["linear_attn_config"][key] = [i for i in text["linear_attn_config"][key] if i < 4]
    config["quantization_config"] = None
    with torch.device("meta"):
        model = transformers.Glm5NextForConditionalGeneration(Glm5NextConfig.from_dict(config))
    selected = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    reader = DictReader({name: torch.empty(1, 1, 1) for name in selected})
    failures = []
    for name in model.state_dict():
        try:
            load_hf_tensor(name, reader, model.config)
        except KeyError as exc:
            failures.append(str(exc))
    assert not failures, "Unmapped published weights: " + ", ".join(failures)

    exported = {published for name, tensor in model.state_dict().items() for published, _ in export_hf_tensor(name, tensor)}
    expected = {name for name in selected if not name.endswith("_scale_inv") and (".layers." not in name or any(f".layers.{i}." in name for i in range(4))) and not name.startswith("model.language_model.layers.45.")}
    assert exported == expected, f"Missing exports: {sorted(expected - exported)[:10]}; extra: {sorted(exported - expected)[:10]}"
