"""CPU checks for the Megatron PP/EP/CP actor's pure contracts."""

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from slime_plugins.models.glm5_next.contract import hf_weight_name, packed_intervals
from slime_plugins.models.glm5_next.parallel_model import (
    Glm5NextLocalExperts,
    forward_packed_moe_layer,
    gather_cp_sequence,
    validate_parallelism,
)
from slime_plugins.models.glm5_next.weight_mapping import checkpoint_name, export_hf_tensor


def _config():
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            mlp_layer_types=["dense"] * 3 + ["sparse"],
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            n_routed_experts=8,
            moe_intermediate_size=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
        ),
        quantization_config=None,
    )


def _args():
    return SimpleNamespace(
        num_layers=4,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        num_experts=8,
        moe_ffn_hidden_size=4,
        moe_shared_expert_intermediate_size=4,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0,
        moe_router_topk_scaling_factor=2.5,
        expert_tensor_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        expert_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=None,
        allgather_cp=True,
    )


def test_pp_ep_cp_allowed_but_tp_rejected():
    args = _args()
    validate_parallelism(args, _config())
    args.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP=1"):
        validate_parallelism(args, _config())
    args.tensor_model_parallel_size = 1
    args.allgather_cp = False
    with pytest.raises(ValueError, match="allgather-cp"):
        validate_parallelism(args, _config())


def test_router_layout_must_match_published_weights():
    args = _args()
    args.moe_router_topk_scaling_factor = 1.0
    with pytest.raises(ValueError, match="moe_router_topk_scaling_factor"):
        validate_parallelism(args, _config())


def test_global_parameter_names_and_packed_offsets():
    assert hf_weight_name("module.module.hf_model.model.language_model.layers.3.self_attn.q_a_proj.weight") == ("model.language_model.layers.3.self_attn.q_a_proj.weight")
    assert packed_intervals([0, 2, 5], 5) == [(0, 2), (2, 5)]
    with pytest.raises(ValueError):
        packed_intervals([0, 2, 2, 5], 5)


def test_local_experts_use_global_ids_and_router_weights(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 1)
    tp_group = object()
    dp_group = object()
    config = SimpleNamespace(
        hidden_size=8,
        moe_ffn_hidden_size=4,
        glm53_swiglu_limit=10.0,
        expert_model_parallel_size=2,
    )
    experts = Glm5NextLocalExperts(
        2,
        config,
        SimpleNamespace(ep=object(), expt_tp=tp_group, expt_dp=dp_group),
    )
    assert list(experts.local_experts) == ["2", "3"]
    assert all(parameter.allreduce is False for parameter in experts.parameters())
    hidden = torch.randn(3, 8)
    weights = torch.tensor([0.2, 0.5, 0.8])
    output, bias = experts(hidden, torch.tensor([2, 1]), weights)
    expected = torch.cat(
        [
            experts.local_experts["2"](hidden[:2]) * weights[:2, None],
            experts.local_experts["3"](hidden[2:]) * weights[2:, None],
        ]
    )
    assert bias is None
    torch.testing.assert_close(output, expected)

    calls = []

    def fake_make_sharded(state_dict, prefix, **kwargs):
        calls.append((prefix, kwargs["tp_group"], kwargs["dp_cp_group"]))
        return {f"{prefix}{name}": tensor for name, tensor in state_dict.items()}

    monkeypatch.setattr(
        "megatron.core.transformer.utils.make_sharded_tensors_for_checkpoint",
        fake_make_sharded,
    )
    sharded = experts.sharded_state_dict(prefix="experts.")
    assert "experts.local_experts.2.gate_proj.weight" in sharded
    assert "experts.local_experts.3.down_proj.weight" in sharded
    assert calls == [
        ("experts.local_experts.2.", tp_group, dp_group),
        ("experts.local_experts.3.", tp_group, dp_group),
    ]


def test_megatron_moe_names_export_as_released_checkpoint_names():
    prefix = "model.language_model.layers.3.mlp"
    cases = {
        f"{prefix}.routed_moe.router.weight": f"{prefix}.gate.weight",
        f"{prefix}.routed_moe.router.expert_bias": f"{prefix}.gate.e_score_correction_bias",
        f"{prefix}.routed_moe.experts.local_experts.72.gate_proj.weight": f"{prefix}.experts.72.gate_proj.weight",
        f"{prefix}.routed_moe.experts.local_experts.72.down_proj.weight": f"{prefix}.experts.72.down_proj.weight",
    }
    for internal, published in cases.items():
        assert checkpoint_name(internal) == published
        assert export_hf_tensor(internal, torch.randn(2, 2))[0][0] == published


def test_expert_checkpoint_replica_ids_use_expert_data_parallel_rank():
    from types import SimpleNamespace

    from slime_plugins.models.glm5_next.parallel_model import use_expert_data_parallel_replica_ids

    expert = SimpleNamespace(replica_id=(0, 0, 7))
    shared = SimpleNamespace(replica_id=(0, 0, 7))
    state = {
        "model.layers.3.mlp.routed_moe.experts.local_experts.252.gate_proj.weight": expert,
        "model.layers.3.mlp.shared_experts.gate_proj.weight": shared,
    }

    assert use_expert_data_parallel_replica_ids(state, expert_dp_rank=2) is state
    assert expert.replica_id == (0, 0, 2)
    assert shared.replica_id == (0, 0, 7)


def test_packed_moe_layer_dispatches_once_for_different_length_sequences():
    class IdentityHyperConnection(torch.nn.Module):
        def forward(self, hidden):
            batch, sequence, streams, _ = hidden.shape
            post = torch.ones((batch, sequence, streams), dtype=hidden.dtype)
            comb = torch.eye(streams, dtype=hidden.dtype).expand(batch, sequence, -1, -1)
            return post, comb, hidden.mean(dim=2)

    class Attention(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states + 1, None, torch.zeros(
                (*hidden_states.shape[:2], 1), dtype=torch.int32
            )

    class CountingMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, hidden):
            self.calls.append(hidden.shape)
            return hidden * 2

    layer = SimpleNamespace(
        block_type="deepseek_sparse_attention",
        attn_hc=IdentityHyperConnection(),
        input_layernorm=torch.nn.Identity(),
        self_attn=Attention(),
        ffn_hc=IdentityHyperConnection(),
        post_attention_layernorm=torch.nn.Identity(),
        mlp=CountingMoe(),
    )
    sequences = [torch.ones(1, 2, 2, 4), torch.ones(1, 3, 2, 4)]
    outputs, topk = forward_packed_moe_layer(layer, sequences, [None, None])

    assert layer.mlp.calls == [torch.Size([1, 5, 4])]
    assert [output.shape for output in outputs] == [torch.Size([1, 2, 2, 4]), torch.Size([1, 3, 2, 4])]
    assert all(indices is not None for indices in topk)


def _cp_worker(rank: int, rendezvous: str):
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        local = torch.tensor([[float(rank + 1)]], requires_grad=True)
        full = gather_cp_sequence(local, torch.distributed.group.WORLD)
        torch.testing.assert_close(full, torch.tensor([[1.0], [2.0]]))
        full[rank].square().sum().backward()
        torch.testing.assert_close(local.grad, torch.tensor([[2.0 * (rank + 1)]]))
    finally:
        torch.distributed.destroy_process_group()


def test_cp_gather_propagates_gradients_across_ranks(tmp_path: Path):
    mp.spawn(_cp_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
