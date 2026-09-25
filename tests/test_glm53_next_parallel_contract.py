"""CPU checks for the Megatron PP/EP/CP actor's pure contracts."""

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from slime_plugins.models.glm5_next.contract import hf_weight_name, packed_intervals
from slime_plugins.models.glm5_next.kernels import (
    GLM53_KERNEL_BACKENDS,
    _ascendc_causal_conv1d,
    _triton_causal_conv1d,
    load_kda_kernel,
)
from slime_plugins.models.glm5_next.parallel_model import (
    Glm5NextLocalExperts,
    Glm5NextMegatronModel,
    _causal_kda_decay_mask,
    configure_npu_attention,
    forward_packed_moe_layer,
    gather_cp_sequence,
    make_pipeline_payload,
    stable_chunk_kimi_delta_attention,
    validate_parallelism,
)
from slime_plugins.models.glm5_next.weight_mapping import checkpoint_name, export_hf_tensor
from slime_plugins.models.glm5_next.validation import alternating_group_reward


def _config(num_layers=4):
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            layer_types=["deepseek_sparse_attention" if (layer + 1) % 4 == 0 else "linear_attention" for layer in range(num_layers)],
            mlp_layer_types=["dense" if layer < 3 else "sparse" for layer in range(num_layers)],
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
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        pipeline_model_parallel_layout=None,
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


def test_eight_layer_pp2_cp2_ep4_layout_is_accepted():
    args = _args()
    args.num_layers = 8
    validate_parallelism(args, _config(num_layers=8))


def test_45_layer_uneven_pp4_layout_is_accepted():
    args = _args()
    args.num_layers = 45
    args.pipeline_model_parallel_size = 4
    args.pipeline_model_parallel_layout = "Et*12|t*11|t*11|t*11L"
    validate_parallelism(args, _config(num_layers=45))

    args.pipeline_model_parallel_layout = None
    args.decoder_first_pipeline_num_layers = 15
    args.decoder_last_pipeline_num_layers = 0
    validate_parallelism(args, _config(num_layers=45))

    args.decoder_last_pipeline_num_layers = 1
    with pytest.raises(ValueError, match="equal positive layer count.*middle"):
        validate_parallelism(args, _config(num_layers=45))

    args.decoder_last_pipeline_num_layers = -1
    with pytest.raises(ValueError, match="last pipeline stage"):
        validate_parallelism(args, _config(num_layers=45))


def test_pipeline_payload_owns_storage_and_preserves_autograd():
    local = torch.randn(1, 2, 4, 3, requires_grad=True)
    payload = make_pipeline_payload(local, hc_mult=4, hidden_size=3)

    assert payload.shape == (2, 1, 12)
    assert payload._base is None
    payload.sum().backward()
    assert torch.equal(local.grad, torch.ones_like(local))


def test_pipeline_model_declares_untied_embeddings():
    assert Glm5NextMegatronModel.share_embeddings_and_output_weights is False


def test_npu_attention_uses_memory_efficient_sdpa():
    config = SimpleNamespace(_attn_implementation="eager")
    configure_npu_attention(config)
    assert config._attn_implementation == "sdpa"


def test_validation_reward_produces_nonzero_advantages_per_group():
    samples = [SimpleNamespace(index=index) for index in range(8)]

    async def collect_rewards():
        return await asyncio.gather(*(alternating_group_reward(None, sample) for sample in samples))

    rewards = asyncio.run(collect_rewards())
    assert rewards == [0.0, 1.0] * 4


def test_kda_decay_masks_future_exponents_before_exp():
    g = torch.full((1, 1, 1, 64, 1), -5.0).cumsum(dim=-2)

    decay = _causal_kda_decay_mask(g, chunk_size=64)

    assert torch.isfinite(decay).all()
    assert decay[0, 0, 0, 0, 63, 0] == 1
    torch.testing.assert_close(decay[0, 0, 0, 63, 0, 0], torch.exp(torch.tensor(-315.0)))


def test_stable_eager_kda_backward_is_finite_for_strong_decay():
    query = torch.randn(1, 64, 1, 2, requires_grad=True)
    key = torch.randn(1, 64, 1, 2, requires_grad=True)
    value = torch.randn(1, 64, 1, 2, requires_grad=True)
    g = torch.full((1, 64, 1, 2), -5.0, requires_grad=True)
    beta = torch.sigmoid(torch.randn(1, 64, 1, requires_grad=True))

    output, _ = stable_chunk_kimi_delta_attention(query, key, value, g, beta, chunk_size=64)
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert all(torch.isfinite(tensor.grad).all() for tensor in (query, key, value, g))


def test_glm53_kernel_backend_contract_and_eager_fallback():
    assert GLM53_KERNEL_BACKENDS == ("ascendc", "triton", "eager")
    eager = object()
    assert load_kda_kernel("eager", eager) is eager
    with pytest.raises(ValueError, match="Unknown GLM-5.3 KDA backend"):
        load_kda_kernel("unknown", eager)


def test_triton_causal_conv1d_layout_adapter():
    hidden = torch.randn(2, 12, 5)
    weight = torch.randn(12, 3)

    def fake_operator(*, x, weight, bias, activation):
        assert x.shape == (2, 5, 12)
        assert weight.shape == (3, 12)
        assert bias is None
        assert activation == "silu"
        return x * 2, None

    output = _triton_causal_conv1d(hidden, weight, None, "silu", fake_operator)
    torch.testing.assert_close(output, hidden * 2)


def test_ascendc_causal_conv1d_splits_qkv_and_restores_layout():
    hidden = torch.randn(2, 24, 5)
    weight = torch.randn(24, 3)
    calls = []

    def fake_operator(*, x, weight, H, bias, activation):
        calls.append((x.shape, weight.shape, H, bias, activation))
        head_dim = x.shape[-1] // H
        return x.reshape(x.shape[0], x.shape[1], H, head_dim).transpose(1, 2), None

    output = _ascendc_causal_conv1d(
        hidden,
        weight,
        bias=None,
        activation="silu",
        num_heads=2,
        operator=fake_operator,
    )
    torch.testing.assert_close(output, hidden)
    assert calls == [((2, 5, 8), (8, 3), 2, None, "silu")] * 3


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
        recompute_granularity="full",
    )
    experts = Glm5NextLocalExperts(
        2,
        config,
        SimpleNamespace(ep=object(), expt_tp=tp_group, expt_dp=dp_group),
    )
    assert list(experts.local_experts) == ["2", "3"]
    assert all(parameter.allreduce is False for parameter in experts.parameters())
    experts.expert_token_chunk_size = 1
    hidden = torch.randn(3, 8, requires_grad=True)
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
    output.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()

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


def test_packed_moe_layer_dispatches_synchronized_chunks_for_different_length_sequences(monkeypatch):
    class IdentityHyperConnection(torch.nn.Module):
        def forward(self, hidden):
            batch, sequence, streams, _ = hidden.shape
            post = torch.ones((batch, sequence, streams), dtype=hidden.dtype)
            comb = torch.eye(streams, dtype=hidden.dtype).expand(batch, sequence, -1, -1)
            return post, comb, hidden.mean(dim=2)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lengths = []

        def forward(self, hidden_states, **kwargs):
            self.lengths.append(hidden_states.shape[1])
            return hidden_states + 1, None, torch.zeros((*hidden_states.shape[:2], 1), dtype=torch.int32)

    class CountingMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, hidden):
            self.calls.append(hidden.shape)
            # Square forces non-reentrant checkpoint to replay the block;
            # multiplying by a constant needs no saved forward tensor.
            return hidden.square()

    layer = SimpleNamespace(
        block_type="deepseek_sparse_attention",
        attn_hc=IdentityHyperConnection(),
        input_layernorm=torch.nn.Identity(),
        self_attn=Attention(),
        ffn_hc=IdentityHyperConnection(),
        post_attention_layernorm=torch.nn.Identity(),
        mlp=CountingMoe(),
    )
    ep_group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fake_all_reduce(length, op, group):
        assert group is ep_group
        assert op is torch.distributed.ReduceOp.MAX
        length.fill_(6)  # Simulate a peer with one more packed token.

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    sequences = [torch.ones(1, 2, 2, 4, requires_grad=True), torch.ones(1, 3, 2, 4, requires_grad=True)]
    outputs, topk = forward_packed_moe_layer(
        layer,
        sequences,
        [None, None],
        recompute=True,
        moe_dispatch_token_chunk_size=3,
        expert_parallel_group=ep_group,
    )
    sum(output.sum() for output in outputs).backward()

    assert layer.mlp.calls == [
        torch.Size([1, 3, 4]),
        torch.Size([1, 3, 4]),
        torch.Size([1, 3, 4]),
        torch.Size([1, 3, 4]),
    ]
    assert sorted(layer.self_attn.lengths) == [2, 2, 3, 3]
    assert [output.shape for output in outputs] == [torch.Size([1, 2, 2, 4]), torch.Size([1, 3, 2, 4])]
    assert all(sequence.grad is not None and torch.isfinite(sequence.grad).all() for sequence in sequences)
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


def test_deferred_lm_head_matches_full_sequence_log_probs_and_gradients(monkeypatch):
    pytest.importorskip("megatron")
    from slime.backends.megatron_utils import loss as loss_module

    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)
    args = SimpleNamespace(
        rollout_temperature=1.25,
        log_probs_chunk_size=1,
        glm53_lm_head_vocab_chunk_size=3,
        entropy_coef=0.0,
        allgather_cp=False,
    )
    tokens = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])]
    total_lengths = [4, 3]
    response_lengths = [2, 1]

    hidden = torch.randn(1, sum(total_lengths), 5, requires_grad=True)
    output_layer = torch.nn.Linear(5, 11, bias=False)
    output_layer.weight.main_grad = torch.zeros_like(output_layer.weight)
    _, deferred = loss_module.get_deferred_log_probs_and_entropy(
        hidden,
        output_layer,
        args=args,
        unconcat_tokens=tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )
    deferred_log_probs = torch.cat(deferred["log_probs"])
    deferred_entropy = torch.cat(deferred["entropy"])
    deferred_log_probs.sum().backward()
    hidden_grad = hidden.grad.clone()
    assert output_layer.weight.grad is None
    assert output_layer.weight.grad_added_to_main_grad
    weight_grad = output_layer.weight.main_grad.clone()

    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_layer = torch.nn.Linear(5, 11, bias=False)
    reference_layer.weight.data.copy_(output_layer.weight.data)
    _, reference = loss_module.get_log_probs_and_entropy(
        reference_layer(reference_hidden),
        args=args,
        unconcat_tokens=tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
    )
    reference_log_probs = torch.cat(reference["log_probs"])
    reference_entropy = torch.cat(reference["entropy"])
    reference_log_probs.sum().backward()

    torch.testing.assert_close(deferred_log_probs, reference_log_probs)
    torch.testing.assert_close(deferred_entropy, reference_entropy)
    torch.testing.assert_close(hidden_grad, reference_hidden.grad)
    torch.testing.assert_close(weight_grad, reference_layer.weight.grad)
