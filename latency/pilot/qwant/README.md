# Qwant

Benchmark current model latency on non-optimized runtime (without CPU optimization).

Baseline latency for model is 60.80ms. We optimize as follows:
- ONNX model yields 16ms
- Optimized ONNX model yields 14ms
- Quantized optimized ONNX model yields 9ms
> Using the following prompt: "Paris est la capitale de la [MASK]."

## Next steps

**The current performance is 80ms for 64 characters, need to get a 10x improvement on this.**

The model has been benchmarked. We'd need to finetune it to downstream task to benchmark it's performance. Could explore the following steps:
- finetune model to FQuAD and PIAF_dev tasks
- hook it onto a low-latency endpoint + add python api
- send to christophe servan for testing + feedback

Try these recipes:
- optimize, quantize
- optimize, prune, quantize
- optimize, prune + quantize aware training

## Deployment platform

Multiple platforms need to be considered for deployment:
- Cloudflare Workers (if we can 1) get onnx runtime running, 2) shrink image down to a small level)
- AWS Lambda @ Edge