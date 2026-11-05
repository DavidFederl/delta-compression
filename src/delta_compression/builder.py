from functools import partial
from typing import Callable, Self

from torch import Tensor

from .base import DeltaCompression, validate_delta_width
from .consecutive_delta import ConsecutiveDelta
from .fixed_reference_delta import FixedReferenceDelta


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
        self.compression_fn = lambda delta, delta_width: delta.clamp(
            -(2 ** (delta_width - 1)), 2 ** (delta_width - 1) - 1
        )
        return self

    def cut(self) -> Self:
        self.compression_fn = lambda delta, delta_width: (
            (delta + 2 ** (delta_width - 1)) % (2**delta_width) - 2 ** (delta_width - 1)
        )
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
