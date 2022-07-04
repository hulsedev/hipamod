# Implementation of the Optimal BERT Surgeon

Neural Magic recently released a paper describing an extension of the Brain Surgeon method for pruning deep neural networks. There implementation leverages an efficient approximation of the Inverse Fisher Information matrix (which describes the loss's Hessian).

Note that there benchmarks indicate significant speedups using only unstructured pruning and the deepsparse engine. Curious to see how these perform on the ONNX runtime.

We make our own implementation of the Optimal BERT Surgeon and look onto applying it to other models.

## Tasks
- [ ] Reproduce baselines from oBERT paper (lottery-ticket, movement pruning)
- [ ] Benchmark performance using ONNX runtime with pruned models

## Goals
- [ ] Reproduction of the presented results, by achieving similar performance
- [ ] Understanding of the underlying bottlenecks for efficient processing of sparse tensors on commodity hardware
- [ ] Suggestions for efficient extension of this method to larger models, and other architectures (generalization)