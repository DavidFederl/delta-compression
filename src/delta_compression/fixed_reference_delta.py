from typing import Callable

from torch import Tensor

from .base import DeltaCompression


class FixedReferenceDelta(DeltaCompression):
    def __init__(
        self,
        delta_width: int,
        reference_value_fn: Callable[[Tensor], Tensor],
        compression_fn: Callable[[Tensor, int], Tensor],
    ):
        super().__init__(delta_width)
        self.reference_value_fn = reference_value_fn
        self.compression_fn = compression_fn

    def _calculate_reference_value(self, input: Tensor) -> Tensor:
        return self.reference_value_fn(input)

    def _calculate_delta(self, input: Tensor, reference_value: Tensor) -> Tensor:
        return input - reference_value

    def _compress_delta(self, delta: Tensor) -> Tensor:
        return self.compression_fn(delta, self.delta_width)

    def _reverse_delta(self, reference_value: Tensor, delta: Tensor) -> Tensor:
        return reference_value + delta
