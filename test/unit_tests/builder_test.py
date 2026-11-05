import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from delta_compression import DeltaCompressionBuilder
from delta_compression.consecutive_delta import ConsecutiveDelta
from delta_compression.fixed_reference_delta import FixedReferenceDelta

values = st.integers(min_value=-10_000, max_value=10_000)
widths = st.integers(min_value=2, max_value=16)
quantizers = st.sampled_from(["clamp", "cut"])
kinds = st.sampled_from(["consecutive", "fixed_reference"])


@st.composite
def sequences(draw, min_size: int = 1) -> torch.Tensor:
    return torch.tensor(
        draw(st.lists(values, min_size=min_size, max_size=64)), dtype=torch.int32
    )


def first(data: torch.Tensor) -> torch.Tensor:
    return data[0]


def configure(kind: str, quantizer: str, width: int) -> DeltaCompressionBuilder:
    builder = DeltaCompressionBuilder()
    if kind == "consecutive":
        builder.consecutive()
    else:
        builder.fixed_reference(first)
    builder.delta_bit_width(width)
    getattr(builder, quantizer)()
    return builder


def fits_signed(delta: torch.Tensor, width: int) -> bool:
    return bool(
        torch.all(delta >= -(2 ** (width - 1)))
        and torch.all(delta <= 2 ** (width - 1) - 1)
    )


@given(quantizer=quantizers, width=widths)
def test_setters_return_builder(quantizer, width):
    builder = DeltaCompressionBuilder()
    assert builder.consecutive() is builder
    assert builder.fixed_reference(first) is builder
    assert builder.delta_bit_width(width) is builder
    assert getattr(builder, quantizer)() is builder


@given(quantizer=quantizers, width=widths)
def test_build_consecutive_type_and_width(quantizer, width):
    codec = configure("consecutive", quantizer, width).build_delta()
    assert isinstance(codec, ConsecutiveDelta)
    assert codec.delta_width == width


@given(quantizer=quantizers, width=widths)
def test_build_fixed_reference_type_and_width(quantizer, width):
    codec = configure("fixed_reference", quantizer, width).build_delta()
    assert isinstance(codec, FixedReferenceDelta)
    assert codec.delta_width == width


@given(width=widths)
def test_fixed_reference_uses_given_reference_function(width):
    codec = (
        DeltaCompressionBuilder()
        .fixed_reference(lambda data: data.min(dim=0).values)
        .delta_bit_width(width)
        .clamp()
        .build_delta()
    )
    data = torch.tensor([4, -2, 9], dtype=torch.int32)
    ref, _ = codec.compress(data)
    assert ref == -2


@given(kind=kinds, width=widths, data=sequences())
def test_clamp_delta_in_signed_range(kind, width, data):
    codec = configure(kind, "clamp", width).build_delta()
    _, delta = codec.compress(data)
    assert fits_signed(delta, width)


@given(kind=kinds, width=widths, data=sequences())
def test_cut_delta_in_signed_range(kind, width, data):
    codec = configure(kind, "cut", width).build_delta()
    _, delta = codec.compress(data)
    assert fits_signed(delta, width)


@given(kind=kinds, quantizer=quantizers, width=widths, data=sequences())
def test_roundtrip_lossless_when_deltas_fit(kind, quantizer, width, data):
    codec = configure(kind, quantizer, width).build_delta()
    true_delta = data[1:] - data[:-1] if kind == "consecutive" else data - data[0]
    ref, delta = codec.compress(data)
    rec = codec.inflate(ref, delta)
    assert torch.equal(rec, data) == fits_signed(true_delta, width)


@given(kind=kinds, width=widths, data=sequences())
def test_cut_roundtrip_modulo_range(kind, width, data):
    codec = configure(kind, "cut", width).build_delta()
    ref, delta = codec.compress(data)
    rec = codec.inflate(ref, delta)
    assert torch.equal(rec % (2**width), data % (2**width))


@given(width=widths, data=sequences())
def test_last_quantizer_wins(width, data):
    builder = configure("consecutive", "clamp", width).cut()
    codec = builder.build_delta()
    _, delta = codec.compress(data)
    expected = configure("consecutive", "cut", width).build_delta().compress(data)[1]
    assert torch.equal(delta, expected)


@given(width=widths)
def test_built_codecs_are_independent(width):
    builder = configure("consecutive", "clamp", width)
    a = builder.build_delta()
    b = builder.delta_bit_width(width + 1).build_delta()
    assert a.delta_width == width
    assert b.delta_width == width + 1


def test_build_without_type_raises():
    builder = DeltaCompressionBuilder().delta_bit_width(8).clamp()
    with pytest.raises(ValueError, match="type"):
        builder.build_delta()


def test_build_without_width_raises():
    builder = DeltaCompressionBuilder().consecutive().clamp()
    with pytest.raises(ValueError, match="width"):
        builder.build_delta()


def test_build_without_compression_raises():
    builder = DeltaCompressionBuilder().consecutive().delta_bit_width(8)
    with pytest.raises(ValueError, match="Compression"):
        builder.build_delta()


@given(width=widths)
def test_builders_do_not_share_state(width):
    configured = configure("consecutive", "clamp", width)
    fresh = DeltaCompressionBuilder()
    assert not hasattr(fresh, "delta_width")
    assert not hasattr(fresh, "delta_compression")
    assert configured.delta_width == width
