# Some Notes

Here are some notes taken along the way of building our module for Highly Parallel Model Deployments.

## Parameters

We start our analysis by focusing on the three parameters: cpu, node, and task count. This follows the setup parameters needed when launching a job on a slurm cluster. We then move on to integrate in our analysis the bandwidth of the system (which can be variable depending on the computing cluster). Finally, we attempt to take into account the price of the deployment of the model (whether the only solution is to rent very expensive instances or running locally). 

## References

- Understanding how academics think about resource allocation when working with institutional clusters, [node allocation at Princeton](https://researchcomputing.princeton.edu/support/knowledge-base/scaling-analysis).
- Running Pytorch on distributed nodes using institutional clusters, [HPC at Princeton](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch#distributed).
- Summary about [model parallelism with Pytorch from Hugging Face](https://huggingface.co/docs/transformers/parallelism).
- [Parallelformers for deploying on multi-GPUs](https://github.com/tunib-ai/parallelformers).
- Benefit from 3D parallelism, described [here](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/).
- Comparison between CPUs and GPUs for [model deployments](https://azure.microsoft.com/en-ca/blog/gpus-vs-cpus-for-deployment-of-deep-learning-models/).
- Distributed training on CPU clusters, [paper from Rice U](https://arxiv.org/pdf/2201.12667v1.pdf).
- Training notes for the BigScience LLM, [on github](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml).
- Pruning pre-trained models, using sparsity, [on arxiv](https://arxiv.org/pdf/2111.05754.pdf).
- Tutorial on [pruning with pytorch](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html).
- Parallelism by [Hugging Face](https://huggingface.co/docs/transformers/v4.16.2/en/parallelism).