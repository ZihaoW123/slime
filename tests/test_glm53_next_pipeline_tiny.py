"""CPU value/gradient equivalence for two GLM-5.3 pipeline stages."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slime_plugins.models.glm5_next import parallel_model


def test_two_stage_pipeline_matches_reference(monkeypatch):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
        Glm5NextTextMoE,
    )

    text = Glm5NextTextConfig(
        vocab_size=128,
        pad_token_id=0,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        index_topk=2,
        index_head_dim=8,
        index_n_heads=4,
        index_kpool=2,
        linear_head_dim=8,
        linear_num_heads=2,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"],
        indexer_types=["full"] * 4,
        use_cache=False,
    )
    vision = Glm5NextVisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        out_hidden_size=64,
        projection_intermediate_size=128,
        num_heads=4,
    )
    config = Glm5NextConfig(text_config=text, vision_config=vision)
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    reference = Glm5NextForConditionalGeneration(config).eval()

    monkeypatch.setattr(parallel_model, "Glm5NextMegatronMoE", lambda cfg, *_: Glm5NextTextMoE(cfg))
    stage_config = SimpleNamespace(hidden_size=text.hidden_size * text.hc_mult)
    first = parallel_model.Glm5NextMegatronModel(stage_config, text, range(2), True, False, None).eval()
    last = parallel_model.Glm5NextMegatronModel(stage_config, text, range(2, 4), False, True, None).eval()
    source = reference.state_dict()
    for stage in (first, last):
        stage.load_state_dict({name: source[name.removeprefix("hf_model.")] for name in stage.state_dict()})

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_context_parallel_group=lambda: None, get_context_parallel_rank=lambda: 0)
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 2, 4, 6]))
    last.set_input_tensor(first(tokens, packed_seq_params=packed))
    actual = last(tokens, packed_seq_params=packed)
    with torch.no_grad():
        expected = torch.cat(
            [reference(input_ids=tokens[:, start:end], use_cache=False).logits for start, end in ((0, 2), (2, 4), (4, 6))],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    first.train()
    last.train()
    first.recompute = True
    last.recompute = True
    last.set_input_tensor(first(tokens, packed_seq_params=packed))
    last(tokens, packed_seq_params=packed).square().mean().backward()
    assert first.hf_model.model.language_model.layers["0"].self_attn.q_proj.weight.grad is not None
    assert last.hf_model.model.language_model.layers["3"].self_attn.q_a_proj.weight.grad is not None
