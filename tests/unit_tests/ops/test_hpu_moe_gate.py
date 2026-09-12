# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.fused_moe import GateLinear

from vllm_gaudi.ops.hpu_moe_gate import HPUGateLinear, get_moe_gate_dtype


@pytest.mark.parametrize(("value", "expected"), [
    (None, torch.float32),
    ("fp32", torch.float32),
    ("float32", torch.float32),
    ("fp16", torch.float16),
    ("float16", torch.float16),
    ("half", torch.float16),
    ("bf16", torch.bfloat16),
    ("bfloat16", torch.bfloat16),
    (" FP16 ", torch.float16),
    ("fp8", torch.float32),
])
def test_get_moe_gate_dtype(monkeypatch, value, expected):
    """Unset means fp32, i.e. upstream behaviour; an unknown value warns and keeps fp32."""
    if value is None:
        monkeypatch.delenv("VLLM_HPU_MOE_GATE_DTYPE", raising=False)
    else:
        monkeypatch.setenv("VLLM_HPU_MOE_GATE_DTYPE", value)

    assert get_moe_gate_dtype() is expected


def test_gate_linear_override_is_registered():
    """PluggableLayer.__new__ dispatches on cls.__name__, so that is the registry key."""
    assert op_registry_oot.get(GateLinear.__name__) is HPUGateLinear
