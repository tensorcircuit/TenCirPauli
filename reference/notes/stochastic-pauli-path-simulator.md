# Stochastic Pauli-path simulator for large-scale quantum optimization

Bibliographic record: Kaining Zhang, Xinbiao Wang, Kunsheng Li, Qixin Zhang, Yuxuan Du, Min-Hsiu Hsieh, and Dacheng Tao, "Stochastic Pauli-path simulator for large-scale quantum optimization," arXiv:2607.17804 [quant-ph], submitted 20 July 2026. The arXiv page lists no journal reference or publication DOI as of 2 August 2026.

## Summary

The paper introduces the stochastic Pauli-path simulator (SPPS). Instead of enumerating or deterministically truncating the branching Pauli propagation tree, SPPS samples one legal branch at each anticommuting Pauli rotation. It assigns the cosine branch probability `q=(|cos(theta)|+a)/(|cos(theta)|+|sin(theta)|+2a)` and the sine branch the complementary probability, with positive smoothing `a` preventing derivative-sensitive branches from receiving zero probability. Importance reweighting makes the sampled value estimator unbiased. Path automatic differentiation (PAD) uses the same sampled paths to produce all active gradient components, with prefix/suffix products replacing the nominal tangent/cotangent score formula near trigonometric zeros. Two independent macro-replicates provide an empirical gradient-error proxy, and per-observable-term sample budgets are doubled until the proxy meets a requested tolerance or reaches a maximum budget.

## Relevance to TenCirPauli

SPPS should be a separate stochastic `value_and_grad` engine, not a mode or threshold of deterministic weight-projected operator propagation. It can reuse deterministic propagation infrastructure: parameter-slot GateTape, packed Pauli words, Clifford conjugation, Pauli-rotation commutation and branch rules, product-state expectation, reusable native handles, deterministic seed plumbing, and coarse Python-to-Rust calls. Its hot representation differs materially: each sample carries one Pauli word, accumulated importance weight, active branch factors, and gradient accumulation state rather than a hash map containing a dynamically expanded Pauli operator.

The natural first implementation scope is circuits interleaving Clifford gates with parameterized Pauli rotations. Static custom PTMs can participate in deterministic propagation, but arbitrary PTMs do not automatically inherit the paper's two-branch trigonometric sampling rule or PAD estimator. SPPS support for a custom gate therefore requires an explicit sampling rule, transition probability, coefficient derivative, and stable local gradient rule; it should not be implied by accepting a forward PTM.

## Claims this paper supports or constrains

The paper supports implementing stochastic single-path propagation as a scalable optimization-oriented complement to deterministic Pauli-weight projection. It also constrains how deterministic truncated gradients should be described: differentiating a weight-truncated recurrence can be mathematically correct for that approximate recurrence while still being biased relative to the exact circuit objective. TenCirPauli documentation and benchmarks must distinguish these two targets.

## Cautions

The unbiasedness guarantee is for the stochastic estimator over the full legal path space and does not imply low variance or efficient sampling for arbitrary circuits. The stated sample complexity depends on circuit depth, smoothing, parameters, and an effective branching factor that may grow exponentially. The paper's A/B stopping statistic is explicitly a practical error proxy rather than a theorem-level confidence interval. Multi-term observables are sampled term by term and combined linearly, which may become expensive for large Hamiltonians unless TenCirPauli later develops justified observable-term sampling or batching. The public anonymous code endpoint exposed only a minimal README during this review, so implementation details must be reconstructed from the paper and independently tested rather than copied from an inspectable reference implementation.
