"""Deterministic validation-only helpers for reduced GLM-5.3 runs."""


async def alternating_group_reward(args, samples, **kwargs):
    """Give every two-sample GRPO group rewards [0, 1].

    The standard reduced checkpoint emits meaningless text, so a task reward
    would be uniformly zero and could not validate backward precision.  Sample
    indices are consecutive within a group in Slime's data source.
    """
    del args, kwargs
    return [float(sample.index % 2) for sample in samples]
