# Factorized Polynomial Attention (FPA)

[](https://github.com/anj1/fpa/actions/workflows/build-and-test.yml)

[cite\_start]This repository contains a PyTorch layer implementing Factorized Polynomial Attention (FPA)[cite: 725, 747]. [cite\_start]FPA is a linear-cost attention mechanism that generalizes Power Attention and offers fine-grained control over its state size[cite: 725, 728]. [cite\_start]The architecture expresses a degree-n polynomial kernel as a product of n inner products in lower-dimensional projected spaces, allowing the state size to be adjusted by tuning the projection dimensions[cite: 726, 727].

For details on the approach, see our post: [https://anj1.github.io/research/posts/fpattention/](https://anj1.github.io/research/posts/fpattention/)

### Features

  - [cite\_start]Exact, deterministic polynomial kernel without Monte-Carlo variance[cite: 725, 737].
  - [cite\_start]Linear scaling with sequence length using an efficient chunked algorithm[cite: 728, 879].
  - [cite\_start]Fine-grained, continuous control over state size by adjusting projection dimensions[cite: 725, 727].
  - [cite\_start]Hardware-friendly implementation with a dual formulation for parallel and recurrent processing[cite: 728, 871, 879].
  - [cite\_start]Support for various special cases and sub-families, including Linear Attention and Power Attention[cite: 728, 754, 755, 855].

## Installation

### From PyPI (Recommended)

```bash
pip install fpa
```

### From Source

Requirements:

  - Python 3.11 or 3.12 (3.13 depends on the upcoming [Triton 3.2 release](https://github.com/triton-lang/triton/issues/5215))
  - CUDA Toolkit 12.4
  - GCC/G++ with C++17 support
  - Linux (Windows/MacOS not supported)

<!-- end list -->

```bash
git clone https://github.com/anj1/fpa.git
cd fpa
pip install -e .
```

All other dependencies (PyTorch, Ninja build system, etc.) will be automatically installed through pip.

## Usage

The main entry point is the `fpa_full` function, which implements Factorized Polynomial Attention. Here's a basic example:

```python
import torch
from fpa.fpa_full import fpa_full

# Create input tensors
batch_size = 2
seq_len = 1024
num_heads = 8
head_dim = 64

Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn_like(Q)
V = torch.randn_like(Q)

# Optional gating tensor
log_G = torch.nn.functional.logsigmoid(
    torch.randn(batch_size, seq_len, num_heads, dtype=torch.float32, device='cuda')
)

# Compute attention
output = fpa_full(
    Q=Q, K=K, V=V, 
    log_G=log_G,          # Optional gating tensor
    degree=2,             # Kernel degree (n)
    branch_dims=[32, 32], # Dimensions of the projected spaces {d_l}. Product must equal head_dim.
    chunk_size=128,       # Size of chunks for processing long sequences
)
```

### Integration with Transformer Models

The package includes a drop-in replacement for standard attention in transformer models.
See `train/model.py` for a complete example of using FPA in a GPT-style model:

```python
from fpa.fpa_full import fpa_full

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... initialization code ...
        
    def forward(self, x):
        # ... projection code ...
        
        # Use FPA instead of standard attention
        y = fpa_full(
            Q=q, K=k, V=v, 
            log_G=log_g,
            degree=self.degree,
            branch_dims=self.branch_dims,
            chunk_size=self.chunk_size
        )
        
        # ... output projection ...
        return y
```

## Development

### Setup

The package uses pip's editable install mode for development. First, activate your Python virtual environment, then:

```bash
# Install base package in editable mode
pip install -e .

# Install development dependencies
pip install psutil
pip install flash_attn==2.7.3 --no-build-isolation
pip install -e .[dev]
```

### Testing & Benchmarking

Run correctness tests:

```bash
pytest
```

Run benchmarks:

```bash
python -m perf.benchmark fwd          // Forward pass
python -m perf.benchmark bwd          // Backward pass
python -m perf.benchmark fwd+bwd      // Forward + backward pass
```

See [benchmark](https://www.google.com/search?q=https://github.com/anj1/fpa/tree/main/perf/README.md) for details.

### Documentation

To view the documentation locally, run:

```bash
pip install mkdocs mkdocs-material
.venv/bin/mkdocs serve -a 0.0.0.0:8000
```

To update it publicly, run:

```bash
mkdocs gh-deploy
```

### Training Example

To immediately see the kernel in action, `cd train` and use:

```bash
# Create the dataset first
python prepare_owt.py

# Single GPU training
python train.py \
  --batch_size=32 \
  --attention_kernel=fpa \
  --degree=2 \
  --branch_dims="[32, 32]" \
  --chunk_size=128 \
  --disable_gating=False

# Multi-GPU training with DDP (example with 4 GPUs)
torchrun --standalone --nproc_per_node=4 train.py \
  --batch_size=32 \
  --attention_kernel=fpa \
  --degree=2 \
  --branch_dims="[32, 32]" \
  --chunk_size=128 \
  --disable_gating=False
```

For distributed training across multiple nodes:

```bash
# On the first (master) node with IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

# On the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

Note: If your cluster does not have Infiniband interconnect, prepend `NCCL_IB_DISABLE=1` to the commands.

## Contributing

We welcome contributions\! Here's how you can help:

### Getting Started

1.  Fork the repository
2.  Create a new branch for your feature/fix: `git checkout -b feature-name`
3.  Install development dependencies: `pip install -e .[dev]`

### Guidelines

  - **Code Style**: Follow PEP 8 for Python code. For CUDA code, follow the existing style in the codebase
  - **Documentation**: Add docstrings to new functions and update README if needed
  - **Testing**: Add tests for new features and ensure all tests pass
  - **Benchmarking**: If your code changes affect performance, delete the `plots/benchmark_results` and rerun some benchmarks with `python -m perf.benchmark fwd+bwd`
  - **Commits**: Write clear, concise commit messages
  - **Performance**: For CUDA kernels, include benchmarks showing performance impact

### Pull Request Process

1.  Update documentation for any new features
2.  Add or update tests as needed
3.  Ensure all tests pass: `pytest`
4.  Run benchmarks if performance-critical code was changed: `python3 -m perf.benchmark fwd+bwd`
5.  Create a Pull Request with a clear description of changes
6.  Wait for review and address any feedback

### Areas for Contribution

  - Performance optimizations for different GPU architectures
  - Documentation improvements
  - Bug fixes
  - Test coverage improvements

For major changes, please open an issue first to discuss what you would like to change.

## Release Process

1.  Update the version in `pyproject.toml`
2.  Run `pytest` and benchmarks if applicable
3.  Run `make release-test` to build & push to Test PyPI for all Python targets
4.  Run `make release` to build & push to PyPI for all Python targets

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nejati2025finely,
  title={Finely Crafted State Spaces for Fast Attention},
  author={Nejati, Alireza},
  journal={arXiv preprint arXiv:2508.04239},
  year={2025}
}
```

## License

Apache 2.0 (see [LICENSE](https://www.google.com/search?q=LICENSE))
