# hipamod (HIghly PArallel MOdel Deployment)

Large Language Models (LLMs) are becoming more common, which is cool, however, they require huge amounts of computing resources to be deployed (see the [metaseq api](https://github.com/facebookresearch/metaseq/blob/main/docs/api.md) for some reference). **What if we could run LLMs using only a few laptops?**

## Goal

This project seeks to determine two things:
- to clarify the problem: how to perform a hardware-dependent scaling analysis for deployment of LLMs?
- to solve it if it is solvable (based on the scaling analysis), otherwise prove that it is not solvable.

>Basically, I'm aware that it's impossible that using CPUs from PCs can compete with TPU/GPU clusters, however, if we're able to achieve even 3% of efficiency compared to the more expensive setups, that still might be very helpful.

**In short, running opt-175B only on distributed CPUs is the mission.**
