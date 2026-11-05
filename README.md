# delta-compression

Delta compression utilities for PyTorch tensors. A tensor is stored as a
reference value plus a sequence of deltas, and each delta is quantized to a fixed
bit width. The original tensor can be reconstructed from the reference and the
deltas.

## Concepts

Two delta schemes are available:

| Scheme            | Reference value                  | Delta                      |
| ----------------- | -------------------------------- | -------------------------- |
| `fixed_reference` | Computed by a function you pass  | `data - reference`         |
| `consecutive`     | First element, `data[0]`         | `data[i + 1] - data[i]`    |

Two quantizers limit each delta to `delta_bit_width` bits (signed):

| Quantizer | Behaviour                                                   |
| --------- | ----------------------------------------------------------- |
| `clamp`   | Saturates deltas to `[-2^(w-1), 2^(w-1) - 1]`               |
| `cut`     | Wraps deltas into the same range (keeps the low `w` bits)   |

Input must be an integer tensor (`int8` to `int32`, `uint8` to `uint32`) with at
least one element. Floats, `bool`, `int64`, `uint64` and complex tensors raise a
`TypeError`. Inputs are widened to `int64`, so unsigned types produce correct
negative deltas and `inflate` returns an `int64` tensor. The delta bit width must
be an integer between 1 and 34, the most that 32 bit input can use.

Reconstruction is exact when every delta fits into the chosen bit width.
Otherwise it is lossy. With `consecutive` and `clamp`, the encoder tracks the
decoder's reconstruction, so an error does not accumulate: it is corrected as
soon as the data moves back into the delta range. A jump larger than the delta
range is still followed with a lag of `2^(w-1)` per step. `cut` is unaffected,
because wrapping already preserves every value modulo `2^w`.

## Project structure

```
src/delta_compression/
    base.py                    DeltaCompression, the abstract base class
    fixed_reference_delta.py   FixedReferenceDelta
    consecutive_delta.py       ConsecutiveDelta
    builder.py                 DeltaCompressionBuilder, the public entry point
test/unit_tests/               pytest and Hypothesis tests, one file per module
devenv.nix                     development environment and check tasks
```

## Usage

```python
import torch

from delta_compression import DeltaCompressionBuilder

data = torch.tensor([5, 7, 4, 4, 9], dtype=torch.int32)

codec = (
    DeltaCompressionBuilder()
    .consecutive()
    .delta_bit_width(8)
    .clamp()
    .build_delta()
)

reference, deltas = codec.compress(data)
# reference: tensor(5)
# deltas:    tensor([ 2, -3,  0,  5])

restored = codec.inflate(reference, deltas)
assert torch.equal(restored, data)
```

For a fixed reference, pass a function that computes the reference from the
data:

```python
codec = (
    DeltaCompressionBuilder()
    .fixed_reference(lambda data: data.min(dim=0).values)
    .delta_bit_width(4)
    .cut()
    .build_delta()
)
```

`build_delta()` raises a `ValueError` if the delta type, the bit width or the
quantizer has not been set.

## Development

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 or newer. A
[devenv](https://devenv.sh) environment is provided and sets everything up.

```sh
uv sync --all-groups     # install dependencies, including dev and test groups
uv run pytest            # run the tests
uv run ruff check        # lint
uv run ruff format       # format
uv run ty check src/     # type check
```

## Contributing

- Branch from `main` and open a pull request.
- Write commit messages in the
  [Conventional Commits](https://www.conventionalcommits.org) format, for example
  `fix: handle empty input`. They are checked with
  [cocogitto](https://docs.cocogitto.io) (`cog check main..`).
- Add or update tests for every behaviour change. Prefer
  [Hypothesis](https://hypothesis.readthedocs.io) properties, such as
  "the round trip is exact if and only if all deltas fit", over single examples.
- Before pushing, make sure `ruff check`, `ruff format --check`, `ty check` and
  `pytest` all pass.
- Keep new delta schemes as subclasses of `DeltaCompression` and expose them
  through `DeltaCompressionBuilder`.

## License

MIT, see [LICENSE](LICENSE).
