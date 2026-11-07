from abc import ABC, abstractmethod

import torch
from torch import Tensor

SUPPORTED_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.uint8,
    torch.uint16,
    torch.uint32,
)
"""Integer dtypes accepted as input, up to 32 bit. They are widened to int64 so that
deltas of unsigned types can be negative and cannot overflow."""

INFLATE_DTYPES = (*SUPPORTED_DTYPES, torch.int64)
"""Dtypes accepted by `inflate`: the supported input dtypes and the int64 tensors
returned by `compress`."""

MAX_DELTA_WIDTH = 34
"""Largest supported delta bit width. Deltas of 32 bit input need at most 33 bits
(34 for a fixed reference outside the input range), so wider deltas are never useful."""


def validate_delta_width(delta_width: int) -> int:
    """Checks that the delta width is an integer between 1 and `MAX_DELTA_WIDTH`.

    Raises:
        TypeError: If the delta width is not an integer.
        ValueError: If the delta width is out of range.
    """
    if isinstance(delta_width, bool) or not isinstance(delta_width, int):
        raise TypeError(f"delta width must be an int, got {type(delta_width).__name__}")
    if not 1 <= delta_width <= MAX_DELTA_WIDTH:
        raise ValueError(
            f"delta width must be between 1 and {MAX_DELTA_WIDTH}, got {delta_width}"
        )
    return delta_width


def _validate_integer(
    tensor: Tensor, name: str, allowed: tuple[torch.dtype, ...]
) -> None:
    if tensor.dtype not in allowed:
        raise TypeError(
            f"{name} must be an integer tensor "
            f"({', '.join(str(d) for d in allowed)}), got {tensor.dtype}"
        )


def _validate_reference(reference_value: Tensor, input: Tensor) -> Tensor:
    if not isinstance(reference_value, Tensor):
        raise TypeError(
            f"reference value must be a tensor, got {type(reference_value).__name__}"
        )
    _validate_integer(reference_value, "reference value", INFLATE_DTYPES)
    try:
        shape = torch.broadcast_shapes(reference_value.shape, input.shape)
    except RuntimeError:
        shape = None
    if shape != input.shape:
        raise ValueError(
            f"reference value of shape {tuple(reference_value.shape)} "
            f"does not broadcast to input of shape {tuple(input.shape)}"
        )
    return reference_value.to(torch.int64)


class DeltaCompression(ABC):
    delta_width: int

    def __init__(self, delta_width: int):
        self.delta_width = validate_delta_width(delta_width)

    @abstractmethod
    def _calculate_reference_value(self, input: Tensor) -> Tensor:
        """Calculates the reference value for the given input.

        Args:
            input (Tensor): The input data.

        Returns:
            Tensor: The calculated reference value.
        """
        ...

    @abstractmethod
    def _calculate_delta(self, input: Tensor, reference_value: Tensor) -> Tensor:
        """Calculates the delta between the input and the reference value.

        Args:
            input (Tensor): The input data.
            reference_value (Tensor): The reference value.
        """
        ...

    @abstractmethod
    def _compress_delta(self, delta: Tensor) -> Tensor:
        """Compresses the delta by quantizing it to the specified delta width.

        Args:
            delta (Tensor): The delta to be compressed.

        Returns:
            Tensor: The compressed delta.
        """
        ...

    def compress(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Compresses the input data by calculating the reference value and delta, then compressing the delta.

        Args:
            input (Tensor): The input data to be compressed.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the reference value and the compressed delta.

        Raises:
            TypeError: If the input or the reference value is not an integer tensor.
            ValueError: If the input is a scalar or empty, or if the reference value
                does not broadcast to the input.
        """
        _validate_integer(input, "input", SUPPORTED_DTYPES)
        if input.ndim == 0 or input.numel() == 0:
            raise ValueError("input must be a non-empty tensor with at least 1 dim")
        input = input.to(torch.int64)
        reference_value = _validate_reference(
            self._calculate_reference_value(input), input
        )
        delta = self._calculate_delta(input, reference_value)
        compressed_delta = self._compress_delta(delta)
        return reference_value, compressed_delta

    def _validate_value_range(self, value_range: tuple[int, int]) -> tuple[int, int]:
        try:
            low, high = value_range
        except (TypeError, ValueError):
            raise TypeError("value range must be a pair (low, high)") from None
        for bound in (low, high):
            if isinstance(bound, bool) or not isinstance(bound, int):
                raise TypeError(
                    f"value range bounds must be ints, got {type(bound).__name__}"
                )
        if low > high:
            raise ValueError(f"value range is empty: low {low} > high {high}")
        if high - low >= 2**self.delta_width:
            raise ValueError(
                f"value range [{low}, {high}] spans more than "
                f"2**{self.delta_width} values and cannot be decoded unambiguously"
            )
        return low, high

    def _validate_delta_range(self, delta: Tensor) -> None:
        if delta.numel() == 0:
            return
        low = -(2 ** (self.delta_width - 1))
        high = 2 ** (self.delta_width - 1) - 1
        if delta.min().item() < low or delta.max().item() > high:
            raise ValueError(
                f"delta values must fit in {self.delta_width} bits "
                f"([{low}, {high}]), got values in "
                f"[{delta.min().item()}, {delta.max().item()}]"
            )

    @abstractmethod
    def _validate_inflate_shapes(self, reference_value: Tensor, delta: Tensor) -> None:
        """Checks that the reference value and the delta fit together.

        Raises:
            ValueError: If the shapes are incompatible.
        """
        ...

    @abstractmethod
    def _reverse_delta(self, reference_value: Tensor, delta: Tensor) -> Tensor:
        """Reconstructs the original input data from the reference value and the decompressed delta.

        Args:
            reference_value (Tensor): The reference value used for reconstruction.
            delta (Tensor): The decompressed delta.

        Returns:
            Tensor: The reconstructed original input data.
        """
        ...

    def inflate(
        self,
        reference_value: Tensor,
        delta: Tensor,
        value_range: tuple[int, int] | None = None,
        check_delta_range: bool = True,
    ) -> Tensor:
        """Decompresses the data by reconstructing the original input from the reference value and the compressed delta.

        Args:
            reference_value (Tensor): The reference value used for decompression.
            delta (Tensor): The compressed delta to be decompressed.
            value_range (tuple[int, int] | None): Inclusive `(low, high)` bounds of the
                original input. Needed to decode deltas that were wrapped by
                `cut_delta`: a wrapped delta only fixes a value modulo
                2**delta_width, so the reconstruction is wrapped into the window of
                that size starting at `low`. The result is exact if
                `high - low < 2**delta_width`. Leave it unset for clamped deltas.
            check_delta_range (bool): Whether to check that all delta values fit in
                `delta_width` bits. Deltas quantized by `clamp_delta` or `cut_delta`
                always do, so the check only catches corrupt or foreign data. It costs
                one pass over the delta. Set this to `False` to skip it, for example
                for a custom compression function that does not respect the width.

        Returns:
            Tensor: The reconstructed original input data as an int64 tensor.

        Raises:
            TypeError: If the reference value or the delta is not an integer tensor.
            TypeError: If the value range is not a pair of integers.
            ValueError: If the shapes of the reference value and the delta do not fit
                together, or if the value range is empty or spans more than
                2**delta_width values, or if `check_delta_range` is set and a delta
                value does not fit in `delta_width` bits.
        """
        _validate_integer(reference_value, "reference_value", INFLATE_DTYPES)
        _validate_integer(delta, "delta", INFLATE_DTYPES)
        self._validate_inflate_shapes(reference_value, delta)
        if check_delta_range:
            self._validate_delta_range(delta)
        original_input = self._reverse_delta(
            reference_value.to(torch.int64), delta.to(torch.int64)
        )
        if value_range is not None:
            low, high = self._validate_value_range(value_range)
            modulus = 2**self.delta_width
            original_input = (original_input - low) % modulus + low
        return original_input
