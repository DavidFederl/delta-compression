import pickle

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from delta_compression import DeltaCompressionBuilder
from delta_compression.base import MAX_DELTA_WIDTH, SUPPORTED_DTYPES

DTYPE_RANGES = {
    torch.int8: (-(2**7), 2**7 - 1),
    torch.int16: (-(2**15), 2**15 - 1),
    torch.int32: (-(2**31), 2**31 - 1),
    torch.uint8: (0, 2**8 - 1),
    torch.uint16: (0, 2**16 - 1),
    torch.uint32: (0, 2**32 - 1),
}
UNSUPPORTED_DTYPES = [
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.bool,
    torch.complex64,
    torch.uint64,
    torch.int64,
]
kinds = st.sampled_from(["consecutive", "fixed_reference"])
quantizers = st.sampled_from(["clamp", "cut"])


def first(data: torch.Tensor) -> torch.Tensor:
    return data[0]


def build(kind: str, quantizer: str, width: int):
    builder = DeltaCompressionBuilder()
    if kind == "consecutive":
        builder.consecutive()
    else:
        builder.fixed_reference(first)
    builder.delta_bit_width(width)
    getattr(builder, quantizer)()
    return builder.build_delta()


@st.composite
def integer_tensors(draw) -> torch.Tensor:
    dtype = draw(st.sampled_from(SUPPORTED_DTYPES))
    low, high = DTYPE_RANGES[dtype]
    data = draw(
        st.lists(st.integers(min_value=low, max_value=high), min_size=1, max_size=32)
    )
    return torch.tensor(data, dtype=dtype)


def test_supported_dtypes_are_all_covered_by_tests():
    assert set(DTYPE_RANGES) == set(SUPPORTED_DTYPES)


@given(kind=kinds, quantizer=quantizers, data=integer_tensors())
def test_roundtrip_for_all_integer_dtypes(kind, quantizer, data):
    codec = build(kind, quantizer, MAX_DELTA_WIDTH)
    ref, delta = codec.compress(data)
    assert torch.equal(codec.inflate(ref, delta), data.to(torch.int64))


@given(kind=kinds, quantizer=quantizers, data=integer_tensors())
def test_compress_output_is_int64(kind, quantizer, data):
    ref, delta = build(kind, quantizer, MAX_DELTA_WIDTH).compress(data)
    assert ref.dtype == torch.int64
    assert delta.dtype == torch.int64


@pytest.mark.parametrize("quantizer", ["clamp", "cut"])
@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_unsigned_input_keeps_negative_deltas(kind, quantizer):
    ref, delta = build(kind, quantizer, 8).compress(
        torch.tensor([10, 3], dtype=torch.uint8)
    )
    expected = [-7] if kind == "consecutive" else [0, -7]
    assert delta.tolist() == expected


@given(
    kind=kinds,
    quantizer=quantizers,
    width=st.integers(min_value=2, max_value=16),
    data=st.lists(st.integers(min_value=0, max_value=255), min_size=2, max_size=32),
)
def test_uint8_matches_int32_result(kind, quantizer, width, data):
    codec = build(kind, quantizer, width)
    as_uint8 = codec.compress(torch.tensor(data, dtype=torch.uint8))
    as_int32 = codec.compress(torch.tensor(data, dtype=torch.int32))
    assert torch.equal(as_uint8[0], as_int32[0])
    assert torch.equal(as_uint8[1], as_int32[1])


@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
@pytest.mark.parametrize("quantizer", ["clamp", "cut"])
@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_non_integer_input_raises(kind, quantizer, dtype):
    data = torch.ones(4).to(dtype)
    with pytest.raises(TypeError, match="integer tensor"):
        build(kind, quantizer, 8).compress(data)


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_empty_input_raises(kind):
    with pytest.raises(ValueError, match="non-empty"):
        build(kind, "clamp", 8).compress(torch.tensor([], dtype=torch.int32))


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_scalar_input_raises(kind):
    with pytest.raises(ValueError, match="non-empty"):
        build(kind, "clamp", 8).compress(torch.tensor(5, dtype=torch.int32))


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_single_element_input_is_supported(kind):
    codec = build(kind, "clamp", 8)
    data = torch.tensor([7], dtype=torch.int32)
    ref, delta = codec.compress(data)
    assert torch.equal(codec.inflate(ref, delta), data)


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_inflate_rejects_float_delta(kind):
    codec = build(kind, "clamp", 8)
    ref, delta = codec.compress(torch.tensor([1, 2, 3], dtype=torch.int32))
    with pytest.raises(TypeError, match="delta"):
        codec.inflate(ref, delta.to(torch.float32))


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_inflate_rejects_float_reference(kind):
    codec = build(kind, "clamp", 8)
    ref, delta = codec.compress(torch.tensor([1, 2, 3], dtype=torch.int32))
    with pytest.raises(TypeError, match="reference_value"):
        codec.inflate(ref.to(torch.float32), delta)


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_inflate_accepts_keyword_arguments(kind):
    codec = build(kind, "clamp", 8)
    data = torch.tensor([1, 2, 4], dtype=torch.int32)
    ref, delta = codec.compress(data)
    assert torch.equal(codec.inflate(reference_value=ref, delta=delta), data)


@given(
    width=st.one_of(
        st.integers(max_value=0),
        st.integers(min_value=MAX_DELTA_WIDTH + 1),
    )
)
def test_builder_rejects_out_of_range_width(width):
    with pytest.raises(ValueError, match="between 1 and"):
        DeltaCompressionBuilder().delta_bit_width(width)


@given(width=st.integers(min_value=1, max_value=MAX_DELTA_WIDTH))
def test_builder_accepts_width_in_range(width):
    assert DeltaCompressionBuilder().delta_bit_width(width).delta_width == width


@pytest.mark.parametrize("width", [8.0, "8", None, True])
def test_builder_rejects_non_int_width(width):
    with pytest.raises(TypeError, match="int"):
        DeltaCompressionBuilder().delta_bit_width(width)


@given(
    kind=kinds,
    width=st.one_of(
        st.integers(max_value=0),
        st.integers(min_value=MAX_DELTA_WIDTH + 1),
    ),
)
def test_codec_constructor_rejects_out_of_range_width(kind, width):
    from delta_compression.consecutive_delta import ConsecutiveDelta
    from delta_compression.fixed_reference_delta import FixedReferenceDelta

    def identity(delta, width):
        return delta

    with pytest.raises(ValueError, match="between 1 and"):
        if kind == "consecutive":
            ConsecutiveDelta(delta_width=width, compression_fn=identity)
        else:
            FixedReferenceDelta(
                delta_width=width, reference_value_fn=first, compression_fn=identity
            )


@given(
    kind=kinds,
    quantizer=quantizers,
    data=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=2),
)
def test_width_one_stays_in_range(kind, quantizer, data):
    _, delta = build(kind, quantizer, 1).compress(torch.tensor(data, dtype=torch.int32))
    assert torch.all((delta >= -1) & (delta <= 0))


@given(
    kind=kinds,
    quantizer=quantizers,
    data=st.lists(
        st.integers(min_value=-(2**31), max_value=2**31 - 1), min_size=2, max_size=8
    ),
)
def test_max_width_is_usable(kind, quantizer, data):
    codec = build(kind, quantizer, MAX_DELTA_WIDTH)
    ref, delta = codec.compress(torch.tensor(data, dtype=torch.int32))
    assert delta.dtype == torch.int64
    assert torch.all(delta >= -(2 ** (MAX_DELTA_WIDTH - 1)))
    assert torch.all(delta <= 2 ** (MAX_DELTA_WIDTH - 1) - 1)


@st.composite
def extreme_pairs(draw) -> torch.Tensor:
    dtype = draw(st.sampled_from(SUPPORTED_DTYPES))
    low, high = (-(2**31), 2**31 - 1) if dtype is torch.int32 else DTYPE_RANGES[dtype]
    return torch.tensor(
        draw(st.lists(st.sampled_from([low, high, 0, low, high]), min_size=2)),
        dtype=dtype,
    )


@given(kind=kinds, quantizer=quantizers, data=extreme_pairs())
def test_33_bit_width_is_always_lossless_for_32_bit_input(kind, quantizer, data):
    codec = build(kind, quantizer, 33)
    ref, delta = codec.compress(data)
    assert torch.equal(codec.inflate(ref, delta), data.to(torch.int64))


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_32_bit_width_is_not_always_lossless(kind):
    data = torch.tensor([-(2**31), 2**31 - 1], dtype=torch.int32)
    codec = build(kind, "clamp", 32)
    ref, delta = codec.compress(data)
    assert not torch.equal(codec.inflate(ref, delta), data.to(torch.int64))


def test_max_delta_width_matches_32_bit_input():
    assert MAX_DELTA_WIDTH == 34


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.uint8])
def test_inflate_widens_small_dtypes_to_int64(kind, dtype):
    codec = build(kind, "clamp", 8)
    high = torch.iinfo(dtype).max
    ref = torch.tensor(high, dtype=dtype)
    delta = torch.tensor([high], dtype=dtype)
    rec = codec.inflate(ref, delta)
    assert rec.dtype == torch.int64
    assert rec[-1].item() == 2 * high


def test_inflate_rejects_mismatched_shapes_for_consecutive():
    codec = build("consecutive", "clamp", 8)
    ref = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="does not match reference value"):
        codec.inflate(ref, torch.zeros(4, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="does not match reference value"):
        codec.inflate(ref, torch.tensor(1))


def test_inflate_rejects_non_broadcastable_reference_for_fixed_reference():
    codec = build("fixed_reference", "clamp", 8)
    ref = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="does not broadcast to delta"):
        codec.inflate(ref, torch.zeros(4, dtype=torch.int64))
    with pytest.raises(ValueError, match="does not broadcast to delta"):
        codec.inflate(
            torch.zeros(2, 4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
        )


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
@pytest.mark.parametrize("quantizer", ["clamp", "cut"])
def test_built_compressors_are_picklable(kind, quantizer):
    codec = build(kind, quantizer, 4)
    data = torch.tensor([0, 5, -9, 100, 3], dtype=torch.int32)
    restored = pickle.loads(pickle.dumps(codec))
    ref, delta = codec.compress(data)
    restored_ref, restored_delta = restored.compress(data)
    assert torch.equal(ref, restored_ref)
    assert torch.equal(delta, restored_delta)


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_inflate_with_value_range_recovers_cut_deltas(kind):
    codec = build(kind, "cut", 4)
    data = torch.tensor([10, 12, 25, 11, 20, 18], dtype=torch.int32)
    ref, delta = codec.compress(data)
    assert not torch.equal(codec.inflate(ref, delta), data.to(torch.int64))
    rec = codec.inflate(ref, delta, value_range=(10, 25))
    assert torch.equal(rec, data.to(torch.int64))
    # A window that does not contain the data wraps values to the wrong place.
    rec = codec.inflate(ref, delta, value_range=(8, 23))
    assert rec[2].item() == 9


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
@given(
    data=st.lists(st.integers(min_value=-40, max_value=40), min_size=1, max_size=32),
    width=st.integers(min_value=3, max_value=8),
)
def test_inflate_with_value_range_is_exact_when_range_fits(kind, data, width):
    codec = build(kind, "cut", width)
    tensor = torch.tensor(data, dtype=torch.int32)
    low, high = min(data), max(data)
    if high - low >= 2**width:
        return
    ref, delta = codec.compress(tensor)
    rec = codec.inflate(ref, delta, value_range=(low, high))
    assert torch.equal(rec, tensor.to(torch.int64))


@pytest.mark.parametrize("kind", ["consecutive", "fixed_reference"])
def test_inflate_rejects_ambiguous_value_range(kind):
    codec = build(kind, "cut", 4)
    ref, delta = codec.compress(torch.tensor([1, 2, 3], dtype=torch.int32))
    with pytest.raises(ValueError, match="cannot be decoded unambiguously"):
        codec.inflate(ref, delta, value_range=(0, 16))
    with pytest.raises(ValueError, match="empty"):
        codec.inflate(ref, delta, value_range=(5, 4))
    with pytest.raises(TypeError, match="pair"):
        codec.inflate(ref, delta, value_range=(1,))
    with pytest.raises(TypeError, match="ints"):
        codec.inflate(ref, delta, value_range=(0.0, 3.0))
