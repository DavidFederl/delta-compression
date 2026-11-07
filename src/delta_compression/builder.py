from functools import partial
from typing import Callable, Self

from torch import Tensor

from .base import DeltaCompression, validate_delta_width
from .consecutive_delta import ConsecutiveDelta
from .fixed_reference_delta import FixedReferenceDelta


def clamp_delta(delta: Tensor, delta_width: int) -> Tensor:
    """Saturates the delta to the signed range of `delta_width` bits."""
    return delta.clamp(-(2 ** (delta_width - 1)), 2 ** (delta_width - 1) - 1)


def cut_delta(delta: Tensor, delta_width: int) -> Tensor:
    """Wraps the delta into the signed range of `delta_width` bits.

    Computes the delta modulo 2**delta_width and maps it into
    [-2**(delta_width - 1), 2**(delta_width - 1) - 1]. In-range deltas are
    unchanged, out-of-range deltas wrap around instead of saturating like
    `clamp_delta`. The offset of 2**(delta_width - 1) shifts the signed range to
    [0, 2**delta_width) so that the modulo (which is non-negative in torch for a
    positive modulus) wraps correctly, and is subtracted again afterwards.

    Example with `delta_width=4`, which covers [-8, 7] (modulus 16):

        delta:     -9   -8    0    7    8    9   20
        result:     7   -8    0    7   -8   -7    4

    For 9: (9 + 8) % 16 - 8 = 1 - 8 = -7. For -9: (-9 + 8) % 16 - 8 = 15 - 8 = 7.
    The wrapped delta is congruent to the original one modulo 16, so decoding is
    only lossless if the decoder wraps as well: pass the bounds of the original input
    as `value_range` to `DeltaCompression.inflate`.
    """
    return (delta + 2 ** (delta_width - 1)) % (2**delta_width) - 2 ** (delta_width - 1)


class DeltaCompressionBuilder:
    delta_compression: partial[DeltaCompression]
    delta_width: int
    compression_fn: Callable[[Tensor, int], Tensor]

    def fixed_reference(
        self, reference_value_function: Callable[[Tensor], Tensor]
    ) -> Self:
        self.delta_compression = partial(
            FixedReferenceDelta, reference_value_fn=reference_value_function
        )
        return self

    def consecutive(self) -> Self:
        self.delta_compression = partial(ConsecutiveDelta)
        return self

    def delta_bit_width(self, bit_width: int) -> Self:
        self.delta_width = validate_delta_width(bit_width)
        return self

    def clamp(self) -> Self:
        self.compression_fn = clamp_delta
        return self

    def cut(self) -> Self:
        self.compression_fn = cut_delta
        return self

    def build_delta(self) -> DeltaCompression:
        if not hasattr(self, "delta_compression"):
            raise ValueError("Delta compression type must be specified")
        if not hasattr(self, "delta_width"):
            raise ValueError("Delta width must be specified")
        if not hasattr(self, "compression_fn"):
            raise ValueError("Compression function must be specified")

        return self.delta_compression(
            delta_width=self.delta_width, compression_fn=self.compression_fn
        )
