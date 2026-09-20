"""Exercise the actual Megatron MoE interface without an Ascend runtime."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime_plugins.models.glm5_next.parallel_model import Glm5NextMegatronMoE


def test_mcore_moe_forward_backward_on_gloo(tmp_path, monkeypatch):
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer.moe import moe_utils
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        mpu.initialize_model_parallel()
        monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
        # This Megatron revision leaves the optional TE symbol undefined when
        # Transformer Engine is absent; its intended fallback is None.
        monkeypatch.setattr(moe_utils, "te_general_gemm", None, raising=False)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=8,
            num_attention_heads=2,
            num_moe_experts=4,
            moe_ffn_hidden_size=4,
            moe_router_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0,
            moe_router_topk_scaling_factor=2.5,
            moe_router_load_balancing_type="none",
            # All-to-all constructs CUDA streams even before forward; the
            # CPU-only contract can still exercise the same expert interface.
            moe_token_dispatcher_type="allgather",
            add_bias_linear=False,
        )
        config.glm53_swiglu_limit = 10.0
        text = Glm5NextTextConfig(
            hidden_size=8,
            moe_intermediate_size=4,
            n_shared_experts=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            routed_scaling_factor=2.5,
        )
        model = Glm5NextMegatronMoE(text, config, 4)
        reference = Glm5NextTextMoE(text)
        with torch.no_grad():
            reference.gate.weight.copy_(model.routed_moe.router.weight)
            reference.gate.e_score_correction_bias.copy_(model.routed_moe.router.expert_bias)
            for index, expert in enumerate(model.routed_moe.experts.local_experts.values()):
                reference.experts.gate_up_proj[index].copy_(
                    torch.cat((expert.gate_proj.weight, expert.up_proj.weight))
                )
                reference.experts.down_proj[index].copy_(expert.down_proj.weight)
            reference.shared_experts.load_state_dict(model.shared_experts.state_dict())
        hidden = torch.randn(1, 3, 8, requires_grad=True)
        output = model(hidden)
        expected = reference(hidden)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
        assert output.shape == hidden.shape
        output.sum().backward()
        assert hidden.grad is not None
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _ep_worker(rank, rendezvous):
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer.moe import moe_utils
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mpu.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
        torch.cuda.current_device = lambda: torch.device("cpu")
        moe_utils.te_general_gemm = None
        config = TransformerConfig(
            num_layers=4,
            hidden_size=8,
            num_attention_heads=2,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=1,
            moe_ffn_hidden_size=4,
            moe_router_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0,
            moe_router_topk_scaling_factor=2.5,
            moe_router_load_balancing_type="none",
            moe_token_dispatcher_type="allgather",
            add_bias_linear=False,
        )
        config.glm53_swiglu_limit = 10.0
        text = Glm5NextTextConfig(hidden_size=8, moe_intermediate_size=4, n_shared_experts=1)
        model = Glm5NextMegatronMoE(text, config, 4)
        assert list(model.routed_moe.experts.local_experts) == [str(rank * 2), str(rank * 2 + 1)]
        hidden = torch.randn(1, 3, 8, requires_grad=True)
        output = model(hidden)
        assert output.shape == hidden.shape
        output.sum().backward()
        assert hidden.grad is not None
        assert all(expert.gate_proj.weight.grad is not None for expert in model.routed_moe.experts.local_experts.values())
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_mcore_ep2_routes_experts_and_backpropagates_on_gloo(tmp_path):
    mp.spawn(_ep_worker, args=(str(tmp_path / "rendezvous-ep2"),), nprocs=2, join=True)
