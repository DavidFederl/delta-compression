import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from delta_compression.fixed_reference_delta import FixedReferenceDelta


def identity(delta: torch.Tensor, width: int) -> torch.Tensor:
    return delta


def clamp(delta: torch.Tensor, width: int) -> torch.Tensor:
    return delta.clamp(-(2 ** (width - 1)), 2 ** (width - 1) - 1)


def cut(delta: torch.Tensor, width: int) -> torch.Tensor:
    return (delta + 2 ** (width - 1)) % (2**width) - 2 ** (width - 1)


def first(data: torch.Tensor) -> torch.Tensor:
    return data[0]


def minimum(data: torch.Tensor) -> torch.Tensor:
    return data.min(dim=0).values


values = st.integers(min_value=-10_000, max_value=10_000)
widths = st.integers(min_value=2, max_value=16)
reference_fns = st.sampled_from([first, minimum])


@st.composite
def sequences(draw, min_size: int = 1) -> torch.Tensor:
    return torch.tensor(
        draw(st.lists(values, min_size=min_size, max_size=64)), dtype=torch.int32
    )


@st.composite
def matrices(draw) -> torch.Tensor:
    rows = draw(st.integers(min_value=1, max_value=16))
    cols = draw(st.integers(min_value=1, max_value=8))
    return torch.tensor(
        draw(
            st.lists(
                st.lists(values, min_size=cols, max_size=cols),
                min_size=rows,
                max_size=rows,
            )
        ),
        dtype=torch.int32,
    )


def make(width: int, compression_fn, reference_fn=first) -> FixedReferenceDelta:
    return FixedReferenceDelta(
        delta_width=width,
        reference_value_fn=reference_fn,
        compression_fn=compression_fn,
    )


def test_delta_example():
    codec = make(8, identity)
    data = torch.tensor([5, 7, 4, 4, 9], dtype=torch.int32)
    ref, delta = codec.compress(data)
    assert ref == 5
    assert torch.equal(delta, torch.tensor([0, 2, -1, -1, 4]))


@given(data=sequences(), width=widths, reference_fn=reference_fns)
def test_reference_comes_from_reference_fn(data, width, reference_fn):
    ref, _ = make(width, identity, reference_fn).compress(data)
    assert torch.equal(ref, reference_fn(data))


@given(data=sequences(), width=widths, reference_fn=reference_fns)
def test_delta_is_difference_to_reference(data, width, reference_fn):
    ref, delta = make(width, identity, reference_fn).compress(data)
    assert delta.shape == data.shape
    assert torch.equal(delta, data - ref)


@given(data=sequences(), width=widths)
def test_first_element_reference_gives_zero_first_delta(data, width):
    _, delta = make(width, identity, first).compress(data)
    assert delta[0] == 0


@given(data=sequences(), width=widths)
def test_minimum_reference_gives_nonnegative_deltas(data, width):
    _, delta = make(width, identity, minimum).compress(data)
    assert torch.all(delta >= 0)


@given(data=sequences(), width=widths, reference_fn=reference_fns)
def test_roundtrip_lossless(data, width, reference_fn):
    codec = make(width, identity, reference_fn)
    ref, delta = codec.compress(data)
    rec = codec.inflate(ref, delta, check_delta_range=False)
    assert torch.equal(rec, data)


@given(data=matrices(), width=widths, reference_fn=reference_fns)
def test_roundtrip_lossless_multidim(data, width, reference_fn):
    codec = make(width, identity, reference_fn)
    ref, delta = codec.compress(data)
    assert delta.shape == data.shape
    rec = codec.inflate(ref, delta, check_delta_range=False)
    assert torch.equal(rec, data)


@given(data=sequences(), width=widths)
def test_clamp_delta_in_signed_range(data, width):
    _, delta = make(width, clamp).compress(data)
    assert torch.all(delta >= -(2 ** (width - 1)))
    assert torch.all(delta <= 2 ** (width - 1) - 1)


@given(data=sequences(), width=widths)
def test_clamp_lossless_iff_all_deltas_fit(data, width):
    codec = make(width, clamp)
    ref, delta = codec.compress(data)
    true_delta = data - ref
    fits = bool(torch.equal(clamp(true_delta, width), true_delta))
    assert torch.equal(codec.inflate(ref, delta), data) == fits


@given(data=sequences(), width=widths)
def test_clamp_error_does_not_accumulate(data, width):
    codec = make(width, clamp)
    ref, delta = codec.compress(data)
    true_delta = data - ref
    error = (codec.inflate(ref, delta) - data).abs()
    assert torch.equal(error, (true_delta - delta).abs())


@given(data=sequences(), width=widths)
def test_cut_delta_in_signed_range(data, width):
    _, delta = make(width, cut).compress(data)
    assert torch.all(delta >= -(2 ** (width - 1)))
    assert torch.all(delta <= 2 ** (width - 1) - 1)


@given(data=sequences(), width=widths)
def test_cut_lossless_iff_all_deltas_fit(data, width):
    codec = make(width, cut)
    ref, delta = codec.compress(data)
    true_delta = data - ref
    fits = bool(torch.equal(cut(true_delta, width), true_delta))
    assert torch.equal(codec.inflate(ref, delta), data) == fits


@given(data=sequences(), width=widths)
def test_cut_roundtrip_modulo_range(data, width):
    codec = make(width, cut)
    ref, delta = codec.compress(data)
    rec = codec.inflate(ref, delta)
    assert torch.equal(rec % (2**width), data % (2**width))


def floor_mean(data: torch.Tensor) -> torch.Tensor:
    return data.sum(dim=0) // data.shape[0]


def maximum(data: torch.Tensor) -> torch.Tensor:
    return data.max(dim=0).values


def zeros(data: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(data)


@given(
    data=matrices(),
    width=widths,
    reference_fn=st.sampled_from([first, minimum, maximum, floor_mean, zeros]),
)
def test_integer_reference_functions_roundtrip(data, width, reference_fn):
    codec = make(34, identity, reference_fn)
    ref, delta = codec.compress(data)
    assert ref.dtype == torch.int64
    assert delta.dtype == torch.int64
    assert torch.equal(codec.inflate(ref, delta), data)


@given(
    data=sequences(),
    dtype=st.sampled_from([torch.int8, torch.int16, torch.int32, torch.uint8]),
)
def test_narrow_integer_reference_is_widened(data, dtype):
    codec = make(34, identity, lambda d: d.min().to(dtype))
    ref, delta = codec.compress(data)
    assert ref.dtype == torch.int64
    assert delta.dtype == torch.int64


@given(
    data=sequences(),
    dtype=st.sampled_from(
        [torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]
    ),
)
def test_non_integer_reference_raises(data, dtype):
    codec = make(8, identity, lambda d: d.min().to(dtype))
    with pytest.raises(TypeError, match="reference value"):
        codec.compress(data)


@given(data=sequences())
def test_float_mean_reference_raises(data):
    codec = make(8, clamp, lambda d: d.float().mean())
    with pytest.raises(TypeError, match="integer tensor"):
        codec.compress(data)


@given(data=sequences())
def test_non_tensor_reference_raises(data):
    codec = make(8, identity, lambda d: int(d[0]))
    with pytest.raises(TypeError, match="must be a tensor"):
        codec.compress(data)


@given(data=sequences())
def test_reference_larger_than_input_raises(data):
    codec = make(8, identity, lambda d: torch.zeros(d.shape[0] + 1, dtype=torch.int64))
    with pytest.raises(ValueError, match="does not broadcast"):
        codec.compress(data)


@given(data=matrices())
def test_reference_with_incompatible_shape_raises(data):
    cols = data.shape[1]
    codec = make(8, identity, lambda d: torch.zeros(cols + 1, dtype=torch.int64))
    with pytest.raises(ValueError, match="does not broadcast"):
        codec.compress(data)
