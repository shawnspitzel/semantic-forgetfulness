# A Brain-Inspired Algorithm That Mitigates Catastrophic Forgetting of Artificial and Spiking Neural Networks
**Source:** https://pubmed.ncbi.nlm.nih.gov/37624895/
**Authors:** Niu, Y. et al.
**Year:** 2023

## Key Findings
- Proposes a biologically-grounded algorithm inspired by hippocampal-neocortical memory consolidation and synaptic tagging-and-capture (STC)
- Introduces a two-stage replay mechanism: a short-term "hippocampal" buffer retains recent experiences; a long-term "neocortical" store receives consolidated representations
- Works across both standard ANNs and spiking neural networks (SNNs), demonstrating broad applicability
- Outperforms EWC, progressive networks, and other continual learning baselines on standard benchmarks (permuted MNIST, split CIFAR)
- Key insight: memory consolidation should happen during idle/offline periods, not during active learning — mirrors biological sleep-dependent consolidation
- Temporal tagging of memories allows selective replay of "surprising" or high-value experiences, reducing redundant rehearsal

## Project Relevance
- The hippocampal buffer / neocortical store architecture is a strong analogy for a two-tier context memory system (working context vs. compressed long-term memory)
- Selective replay based on surprise/novelty is directly applicable to deciding which parts of LLM context to retain vs. compress/discard
- SNN compatibility hints at potential for efficient, sparse memory representations — relevant to token-efficient context management
