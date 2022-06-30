# Qwant

Benchmark current model latency on non-optimized runtime (without CPU optimization).

Baseline latency for model is 60.80ms. We optimize as follows:
- ONNX model yields 16ms
- Optimized ONNX model yields 14ms
- Quantized optimized ONNX model yields 9ms

## Next steps

The model has been benchmarked. We'd need to finetune it to downstream task to benchmark it's performance. Could explore the following steps:
- finetune model to FQuAD and PIAF_dev tasks
- hook it onto a low-latency endpoint + add python api
- send to christophe servan for testing + feedback