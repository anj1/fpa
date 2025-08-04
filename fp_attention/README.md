## Kernel Custom Ops

This directory contains thin wrappers around custom CUDA kernels.
The wrappers are necessary to enable `torch.compile()` tracing, and to
link forward and backward implementations.