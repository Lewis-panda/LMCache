# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for rank-4 sub-paged vLLM attention caches."""

from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from unittest.mock import patch
import sys

# Third Party
import pytest
import torch


LOGICAL_BLOCK_SIZE = 1600
KERNEL_BLOCK_SIZE = 64
NUM_LOGICAL_BLOCKS = 2
HEAD_SIZE = 8


class SpecKind(Enum):
    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW = "sliding_window"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    SINK_FULL_ATTENTION = "sink_full_attention"
    CROSS_ATTENTION = "cross_attention"
    MLA_ATTENTION = "mla_attention"
    MAMBA = "mamba"


@dataclass
class Spec:
    kind: SpecKind
    block_size: int
    page_size_bytes: int
    num_heads: int
    state_content_size_bytes: int
    mamba_cache_mode: str = "align"
    compress_ratio: int = 1


@dataclass
class Group:
    layer_names: list[str]
    kv_cache_spec: Spec


@dataclass
class Config:
    kv_cache_groups: list[Group]
    has_mamba_layers: bool = True


@pytest.fixture(scope="module")
def edits():
    """Import the module with only its vLLM type surface stubbed."""
    stub = ModuleType("vllm.v1.kv_cache_interface")
    stub.KVCacheConfig = Config
    stub.KVCacheSpec = Spec
    stub.KVCacheSpecKind = SpecKind
    stub.get_kv_cache_spec_kind = lambda value: value.kind
    packages = {
        "vllm": ModuleType("vllm"),
        "vllm.v1": ModuleType("vllm.v1"),
        "vllm.v1.kv_cache_interface": stub,
    }
    module_name = "lmcache.integration.vllm.kv_cache_group_edits"
    with patch.dict(sys.modules, packages):
        sys.modules.pop(module_name, None)
        from lmcache.integration.vllm import kv_cache_group_edits

        yield kv_cache_group_edits
    sys.modules.pop(module_name, None)


def attention_spec(
    *,
    num_heads: int = 2,
    page_size_multiplier: int = 1,
    declared_content_multiplier: int = 1,
    compress_ratio: int = 1,
) -> Spec:
    elem_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    content_bytes = 2 * HEAD_SIZE * elem_size
    page_bytes = LOGICAL_BLOCK_SIZE * num_heads * content_bytes
    return Spec(
        kind=SpecKind.FULL_ATTENTION,
        block_size=LOGICAL_BLOCK_SIZE,
        page_size_bytes=page_bytes * page_size_multiplier,
        num_heads=num_heads,
        state_content_size_bytes=content_bytes * declared_content_multiplier,
        compress_ratio=compress_ratio,
    )


def registered_cache(
    layout: str,
    *,
    num_heads: int = 2,
    kernel_block_size: int = KERNEL_BLOCK_SIZE,
    contiguous_nhd_shape: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = LOGICAL_BLOCK_SIZE // kernel_block_size
    pages = NUM_LOGICAL_BLOCKS * ratio
    content_size = 2 * HEAD_SIZE
    flat = torch.arange(
        pages * num_heads * kernel_block_size * content_size,
        dtype=torch.bfloat16,
    )
    if layout == "NHD":
        physical = flat.view(pages, kernel_block_size, num_heads, content_size)
        if contiguous_nhd_shape:
            return physical, physical
        return physical.permute(0, 2, 1, 3), physical
    if layout == "HND":
        physical = flat.view(pages, num_heads, kernel_block_size, content_size)
        return physical, physical
    raise ValueError(layout)


def apply_edit(
    edits,
    cache: torch.Tensor,
    spec: Spec,
    *,
    layout: str,
    has_mamba_layers: bool = True,
) -> torch.Tensor:
    config = Config([Group(["attn.0"], spec)], has_mamba_layers=has_mamba_layers)
    return edits.apply_kv_cache_group_edits(
        config, {"attn.0": cache}, {"kv_layout": layout}
    )["attn.0"]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_rank4_contract_reaches_detector_at_logical_granularity(edits, layout):
    cache, physical = registered_cache(layout)
    viewed = apply_edit(edits, cache, attention_spec(), layout=layout)

    expected_shape = (
        (NUM_LOGICAL_BLOCKS, LOGICAL_BLOCK_SIZE, 1, 4 * HEAD_SIZE)
        if layout == "NHD"
        else (NUM_LOGICAL_BLOCKS, 1, LOGICAL_BLOCK_SIZE, 4 * HEAD_SIZE)
    )
    assert tuple(viewed.shape) == expected_shape
    assert viewed.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert viewed.numel() == cache.numel()
    torch.testing.assert_close(viewed.reshape(-1), physical.reshape(-1))

    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.kv_format import detect_format, get_spec
    from lmcache.v1.gpu_connector.kv_format.detectors import vllm as detector_module

    with patch.object(detector_module, "torch_device_type", "cuda"):
        detected_format, normalized = detect_format(
            [viewed], EngineType.VLLM, {"kv_layout": layout}
        )
    assert get_spec(normalized, detected_format).block_size() == LOGICAL_BLOCK_SIZE


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("num_heads", [1, 2, 32])
def test_rank4_contract_handles_head_cardinality(edits, layout, num_heads):
    cache, physical = registered_cache(layout, num_heads=num_heads)
    viewed = apply_edit(
        edits, cache, attention_spec(num_heads=num_heads), layout=layout
    )
    assert viewed.shape[0] == NUM_LOGICAL_BLOCKS
    token_axis = 1 if layout == "NHD" else 2
    assert viewed.shape[token_axis] == LOGICAL_BLOCK_SIZE
    torch.testing.assert_close(viewed.reshape(-1), physical.reshape(-1))


def test_contiguous_nhd_shape_is_supported(edits):
    cache, physical = registered_cache("NHD", contiguous_nhd_shape=True)
    viewed = apply_edit(edits, cache, attention_spec(), layout="NHD")
    assert tuple(viewed.shape[:3]) == (NUM_LOGICAL_BLOCKS, LOGICAL_BLOCK_SIZE, 1)
    torch.testing.assert_close(viewed.reshape(-1), physical.reshape(-1))


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_each_logical_block_groups_adjacent_kernel_pages(edits, layout):
    cache, physical = registered_cache(layout)
    viewed = apply_edit(edits, cache, attention_spec(), layout=layout)
    ratio = LOGICAL_BLOCK_SIZE // KERNEL_BLOCK_SIZE
    for block in range(NUM_LOGICAL_BLOCKS):
        expected = physical[block * ratio : (block + 1) * ratio]
        torch.testing.assert_close(viewed[block].reshape(-1), expected.reshape(-1))


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_unsubpaged_rank4_cache_passes_through(edits, layout):
    cache, _ = registered_cache(layout, kernel_block_size=LOGICAL_BLOCK_SIZE)
    assert apply_edit(edits, cache, attention_spec(), layout=layout) is cache


def test_declared_compression_passes_through(edits):
    cache, _ = registered_cache("NHD")
    spec = attention_spec(compress_ratio=2)
    assert apply_edit(edits, cache, spec, layout="NHD") is cache


def test_non_hybrid_config_passes_through(edits):
    cache, _ = registered_cache("NHD")
    assert (
        apply_edit(
            edits,
            cache,
            attention_spec(),
            layout="NHD",
            has_mamba_layers=False,
        )
        is cache
    )


def test_missing_layout_hint_fails_loudly(edits):
    cache, _ = registered_cache("NHD")
    config = Config([Group(["attn.0"], attention_spec())])
    with pytest.raises(ValueError, match="Unsupported kv_layout"):
        edits.apply_kv_cache_group_edits(config, {"attn.0": cache}, {})


def test_doubled_page_contract_fails_loudly(edits):
    cache, _ = registered_cache("NHD")
    with pytest.raises(ValueError, match="byte-derived kernel block size"):
        apply_edit(
            edits,
            cache,
            attention_spec(page_size_multiplier=2),
            layout="NHD",
        )


def test_head_count_cannot_mask_a_bad_page_contract(edits):
    """A 32-head axis must not masquerade as a fake 32-token kernel page."""
    cache, _ = registered_cache("NHD", num_heads=32)
    with pytest.raises(ValueError, match="inconsistent with middle axes"):
        apply_edit(
            edits,
            cache,
            attention_spec(num_heads=32, page_size_multiplier=2),
            layout="NHD",
        )


def test_declared_content_mismatch_fails_loudly(edits):
    cache, _ = registered_cache("HND")
    with pytest.raises(ValueError, match="content axis carries"):
        apply_edit(
            edits,
            cache,
            attention_spec(declared_content_multiplier=2),
            layout="HND",
        )


def test_non_dense_kernel_pages_fail_loudly(edits):
    cache, _ = registered_cache("HND")
    strided = cache.repeat_interleave(2, dim=0)[::2]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        apply_edit(edits, strided, attention_spec(), layout="HND")


def test_rank5_legacy_rule_is_unchanged(edits):
    ratio = LOGICAL_BLOCK_SIZE // KERNEL_BLOCK_SIZE
    cache = torch.zeros(
        NUM_LOGICAL_BLOCKS * ratio,
        2,
        KERNEL_BLOCK_SIZE,
        2,
        HEAD_SIZE,
        dtype=torch.bfloat16,
    )
    viewed = apply_edit(edits, cache, attention_spec(), layout="NHD")
    assert tuple(viewed.shape[:3]) == (NUM_LOGICAL_BLOCKS, 2, LOGICAL_BLOCK_SIZE)
    assert viewed.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()


def test_observability_name_is_preserved(edits):
    cache, _ = registered_cache("HND")
    with patch.object(edits.logger, "info") as mock_info:
        apply_edit(edits, cache, attention_spec(), layout="HND")
    assert mock_info.call_args.args[1] == {"subpaged-attention-view": 1}
