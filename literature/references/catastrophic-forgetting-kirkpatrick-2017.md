# Overcoming Catastrophic Forgetting in Neural Networks
**Source:** https://www.pnas.org/doi/10.1073/pnas.1611835114
**Authors:** Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., Hadsell, R.
**Publisher:** PNAS
**Year:** 2017

## Key Findings
- Introduces **Elastic Weight Consolidation (EWC)**: a regularization method that slows down learning on weights important to previously learned tasks
- Importance of each weight is estimated via the diagonal of the **Fisher information matrix**, which approximates the curvature of the loss landscape
- EWC allows a single network to learn multiple tasks sequentially without forgetting prior tasks (demonstrated on permuted MNIST and Atari games)
- Without EWC, networks trained sequentially on two tasks showed near-complete forgetting of the first task (~95%+ performance drop)
- With EWC, performance on prior tasks was largely preserved while new tasks were learned — matching or approaching multi-task training baselines
- Draws explicit analogy to synaptic consolidation in biological brains (e.g., long-term potentiation)

## Project Relevance
- Foundational framework for thinking about "what to forget vs. what to protect" — directly maps to the semantic forgetfulness problem
- Fisher information as a forgetting saliency metric is a tractable approach for weighting which context/knowledge to preserve
- The sequential learning paradigm mirrors how LLMs accumulate context over long conversations — older information is at risk of being overwritten or deprioritized
