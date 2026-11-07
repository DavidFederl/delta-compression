from typing import Callable

from torch import Tensor, cat, empty_like

from .base import DeltaCompression


class ConsecutiveDelta(DeltaCompression):
    def __init__(
        self,
        delta_width: int,
        compression_fn: Callable[[Tensor, int], Tensor],
    ):
        super().__init__(delta_width)
        self.reference_value_fn = lambda input: input[0]
        self.compression_fn = compression_fn

    def _calculate_reference_value(self, input: Tensor) -> Tensor:
        return self.reference_value_fn(input)

    def _calculate_delta(self, input: Tensor, reference_value: Tensor) -> Tensor:
        """Calculates the quantized deltas in a closed loop.

        Each delta is taken against the value the decoder will have reconstructed
        so far, not against the original previous value. A quantization error is
        therefore seen by the next delta and corrected when it fits into the delta
        width, instead of being carried through the cumulative sum.
        """
        deltas = empty_like(input[1:])
        reconstruction = reference_value
        for i in range(1, input.shape[0]):
            delta = self.compression_fn(input[i] - reconstruction, self.delta_width)
            deltas[i - 1] = delta
            reconstruction = reconstruction + delta
        return deltas

    def _compress_delta(self, delta: Tensor) -> Tensor:
        # Already quantized in _calculate_delta, which has to do it inside the loop.
        return delta

    def _validate_inflate_shapes(self, reference_value: Tensor, delta: Tensor) -> None:
        if delta.ndim == 0 or delta.shape[1:] != reference_value.shape:
            raise ValueError(
                f"delta of shape {tuple(delta.shape)} does not match reference "
                f"value of shape {tuple(reference_value.shape)}: expected a delta "
                f"of shape (n - 1, *{tuple(reference_value.shape)})"
            )

    def _reverse_delta(self, reference_value: Tensor, delta: Tensor) -> Tensor:
        return cat(
            [reference_value.unsqueeze(0), reference_value + delta.cumsum(dim=0)]
        )
