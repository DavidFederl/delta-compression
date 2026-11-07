import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from delta_compression.consecutive_delta import ConsecutiveDelta


def identity(delta: torch.Tensor, width: int) -> torch.Tensor:
    return delta


def clamp(delta: torch.Tensor, width: int) -> torch.Tensor:
    return delta.clamp(-(2 ** (width - 1)), 2 ** (width - 1) - 1)


def cut(delta: torch.Tensor, width: int) -> torch.Tensor:
    return (delta + 2 ** (width - 1)) % (2**width) - 2 ** (width - 1)


values = st.integers(min_value=-10_000, max_value=10_000)
widths = st.integers(min_value=2, max_value=16)


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


def test_delta_example():
    codec = ConsecutiveDelta(delta_width=8, compression_fn=identity)
    data = torch.tensor([5, 7, 4, 4, 9], dtype=torch.int32)
    ref, delta = codec.compress(data)
    assert ref == 5
    assert torch.equal(delta, torch.tensor([2, -3, 0, 5]))


@pytest.mark.parametrize("quantizer", [identity, clamp, cut])
def test_single_element_has_empty_delta(quantizer):
    codec = ConsecutiveDelta(delta_width=8, compression_fn=quantizer)
    data = torch.tensor([42], dtype=torch.int32)
    ref, delta = codec.compress(data)
    assert ref == 42
    assert delta.shape == (0,)
    assert delta.dtype == torch.int64
    assert torch.equal(codec.inflate(ref, delta), data.to(torch.int64))


@pytest.mark.parametrize("quantizer", [identity, clamp, cut])
def test_single_row_has_empty_delta(quantizer):
    codec = ConsecutiveDelta(delta_width=8, compression_fn=quantizer)
    data = torch.tensor([[1, -2, 3]], dtype=torch.int32)
    ref, delta = codec.compress(data)
    assert torch.equal(ref, torch.tensor([1, -2, 3]))
    assert delta.shape == (0, 3)
    assert torch.equal(codec.inflate(ref, delta), data.to(torch.int64))


@given(data=sequences(), width=widths)
def test_reference_is_first_element(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=identity)
    ref, _ = codec.compress(data)
    assert ref == data[0]


@given(data=sequences(), width=widths)
def test_delta_is_neighbour_difference(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=identity)
    _, delta = codec.compress(data)
    assert delta.shape[0] == data.shape[0] - 1
    assert torch.equal(delta, data[1:] - data[:-1])


@given(data=sequences(), width=widths)
def test_roundtrip_lossless(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=identity)
    ref, delta = codec.compress(data)
    rec = codec.inflate(ref, delta, check_delta_range=False)
    assert torch.equal(rec, data)


@given(data=matrices(), width=widths)
def test_roundtrip_lossless_multidim(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=identity)
    ref, delta = codec.compress(data)
    assert delta.shape == (data.shape[0] - 1, data.shape[1])
    rec = codec.inflate(ref, delta, check_delta_range=False)
    assert torch.equal(rec, data)


@given(data=sequences(), width=widths)
def test_clamp_delta_in_signed_range(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=clamp)
    _, delta = codec.compress(data)
    assert torch.all(delta >= -(2 ** (width - 1)))
    assert torch.all(delta <= 2 ** (width - 1) - 1)


@given(data=sequences(), width=widths)
def test_clamp_lossless_iff_all_deltas_fit(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=clamp)
    ref, delta = codec.compress(data)
    true_delta = data[1:] - data[:-1]
    fits = bool(torch.equal(clamp(true_delta, width), true_delta))
    assert torch.equal(codec.inflate(ref, delta), data) == fits


@given(data=sequences(), width=widths)
def test_clamp_reconstruction_follows_clamped_deltas(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=clamp)
    ref, delta = codec.compress(data)
    rec = codec.inflate(ref, delta)
    assert rec[0] == data[0]
    assert torch.equal(rec[1:] - rec[:-1], delta)


@given(data=sequences(), width=widths)
def test_cut_delta_in_signed_range(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=cut)
    _, delta = codec.compress(data)
    assert torch.all(delta >= -(2 ** (width - 1)))
    assert torch.all(delta <= 2 ** (width - 1) - 1)


@given(data=sequences(), width=widths)
def test_cut_lossless_iff_all_deltas_fit(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=cut)
    ref, delta = codec.compress(data)
    true_delta = data[1:] - data[:-1]
    fits = bool(torch.equal(cut(true_delta, width), true_delta))
    assert torch.equal(codec.inflate(ref, delta), data) == fits


@given(data=sequences(), width=widths)
def test_cut_roundtrip_modulo_range(data, width):
    codec = ConsecutiveDelta(delta_width=width, compression_fn=cut)
    ref, delta = codec.compress(data)
    rec = codec.inflate(ref, delta)
    assert torch.equal(rec % (2**width), data % (2**width))


def test_clamp_recovers_after_spike():
    codec = ConsecutiveDelta(delta_width=4, compression_fn=clamp)
    data = torch.tensor([0, 100, 0, 0, 0, 0], dtype=torch.int32)
    ref, delta = codec.compress(data)
    error = codec.inflate(ref, delta) - data
    assert error.tolist() == [0, -93, 0, 0, 0, 0]


@given(
    jump=st.integers(min_value=-5_000, max_value=5_000).filter(lambda j: j != 0),
    width=widths,
)
def test_clamp_error_vanishes_once_data_settles(jump, width):
    # After a jump the reconstruction catches up by the largest delta per step,
    # so a constant tail longer than jump / step brings the error back to zero.
    step = 2 ** (width - 1) - 1 if jump > 0 else 2 ** (width - 1)
    tail = -(-abs(jump) // step) + 1
    data = torch.tensor([0] + [jump] * tail, dtype=torch.int32)
    codec = ConsecutiveDelta(delta_width=width, compression_fn=clamp)
    ref, delta = codec.compress(data)
    assert codec.inflate(ref, delta)[-1] == jump


@given(data=sequences(), width=widths)
def test_clamp_error_is_bounded_by_the_overload(data, width):
    # The error of a step is the part of (previous error + step) that does not
    # fit into the delta range, so it never exceeds that sum.
    codec = ConsecutiveDelta(delta_width=width, compression_fn=clamp)
    ref, delta = codec.compress(data)
    error = (data.to(torch.int64) - codec.inflate(ref, delta)).abs()
    steps = (data[1:] - data[:-1]).abs().to(torch.int64)
    assert torch.all(error[1:] <= error[:-1] + steps)
