# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU override for the MoE router gate (``GateLinear``).

Upstream ``GateLinear`` stores its weight in fp32 whenever ``force_fp32_compute``
is requested and no CUDA-specialised router kernel exists.  On HPU that second
condition is always true (the check is gated on ``is_cuda()``), so every model
asking for an fp32 router — Nemotron-H / Nemotron-3, MiniMax-M2 — upcasts its
checkpoint weight to fp32 and then runs an fp32 MME GEMM.

Two measurements on Gaudi3 make that a bad trade for Nemotron-3-Ultra-550B:

* the ``[1,8192]x[8192,512]`` gate GEMM costs 41.8 us/layer in fp32 versus
  21.2 in fp16 and 8.2 in bf16, i.e. **2.0 ms of the 15.1 ms device time of a
  decode step** and 27% of all GEMM time, for a matvec;
* the checkpoint stores ``mixer.gate.weight`` as **bf16**, so the fp32 weight
  carries no information the bf16 one does not.

``VLLM_HPU_MOE_GATE_DTYPE`` selects the gate's storage/compute dtype:

===========  ===============  ==================================================
value        us/layer (G3)    top-22 expert set identical to the exact result
===========  ===============  ==================================================
``fp32``     41.8             100.000%   (default, upstream behaviour)
``fp16``     21.2              99.544%
``bf16``      8.2              94.889%
===========  ===============  ==================================================

``fp16`` is validated end to end on Nemotron-3-Ultra-550B at TP8+EP (GaudiSW
1.24.2-399): the router-gate GEMM goes 2.052 -> 1.076 ms per decode step and
total device busy 15.13 -> 14.17 ms (-6.4%), while gsm8k_platinum 5-shot scores
0.9750 with the **identical failure set** {24, 34, 54, 65, 126} and a generation
agreement (0.635) that sits between two runs of the unmodified stack
(0.630 / 0.665).  It is opt-in rather than the default because MiniMax-M2/M2.5/M3
reach this path too and were **not** measured.

``bf16`` is 2.5x faster again on the gate (measured 0.43 ms/step against fp16's
1.08 and fp32's 2.05) but **failed the same accuracy gate**: gsm8k_platinum drops
0.9750 -> 0.9700 and one prompt that fp16 answers correctly instead collapses into
a verbatim repetition loop of the question, emitting no answer at all.  That is
consistent with bf16 perturbing 20x more routing decisions.  Treat bf16 as a
throughput-only option for a caller who has re-run their own accuracy suite.

Note on the dtypes: fp16 keeps 10 mantissa bits but only a 5-bit exponent, so
unlike bf16 it can overflow on an outlier gate input.  It is safe on Nemotron
because the gate consumes a post-RMSNorm hidden state and the logits are
~N(0, 1.8) by construction (weight ~0.02, K=8192); do not enable fp16 for a gate
whose input is not normalised without re-checking.

The fidelity column isolates GEMM *output* rounding, because the weight itself is
exact in every row (bf16 -> fp16 is lossless at these magnitudes).  No
bf16-in/fp32-out GEMM exists on this stack: ``torch.mm(out_dtype=)`` is not
implemented for HPU and ``torch.mm(..., out=fp32_buffer)`` is rejected by
PyTorch's own meta function under ``torch.compile``.

Router logits are still returned in ``out_dtype`` (fp32), so grouped-topk and
everything downstream are unchanged.
"""

import torch

from vllm.model_executor.layers.fused_moe import GateLinear

from vllm_gaudi import envs
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_GATE_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

_ENV = "VLLM_HPU_MOE_GATE_DTYPE"


def get_moe_gate_dtype() -> torch.dtype:
    """Storage/compute dtype for an fp32-requesting MoE router gate on HPU."""
    raw = envs.VLLM_HPU_MOE_GATE_DTYPE
    dtype = _GATE_DTYPES.get(raw)
    if dtype is None:
        logger.warning("%s=%r is not one of %s; keeping fp32", _ENV, raw, sorted(_GATE_DTYPES))
        return torch.float32
    return dtype


@GateLinear.register_oot
class HPUGateLinear(GateLinear):
    """``GateLinear`` that can keep the router weight at its checkpoint width."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        if force_fp32_compute and params_dtype is None:
            requested = get_moe_gate_dtype()
            if requested is not torch.float32:
                # Neutralise the upstream fp32 upcast: with force_fp32_compute
                # cleared the base class keeps params_dtype, and because none of
                # the specialised router GEMM tiers are reachable on HPU the
                # forward falls through to F.linear in `requested` and casts the
                # result to out_dtype.
                logger.info_once("HPU MoE gate: %s=%s (upstream would use fp32)", _ENV, requested)
                force_fp32_compute = False
                params_dtype = requested

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            out_dtype=out_dtype,
            params_dtype=params_dtype,
            force_fp32_compute=force_fp32_compute,
            prefix=prefix,
        )
