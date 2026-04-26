## Problem statement

Large language models (LLMs) can generate fluent answers that contain false statements (“hallucinations”). In practical use, a system needs a quantitative signal of when an output is likely unreliable so it can warn the user or abstain. A central difficulty is that the correctness of an answer is primarily semantic: different token sequences can express the same meaning, and token-level uncertainty does not reliably track factual correctness.

Semantic uncertainty quantification methods address this by measuring uncertainty in a semantic space rather than in token space. The SeSE framework (Semantic Structural Entropy) is a semantic uncertainty quantification approach for hallucination detection that explicitly models *structure* in semantic spaces. The project artifacts provided here include the SeSE paper (`2511.16275v1.pdf`) and an implementation notebook (`SESE_FIXED_THRESHOLDS (1).ipynb`) that demonstrates short-form uncertainty scoring and claim-level (long-form) hallucination classification.

## Goals and challenges (project enhancement relative to SeSE)

The goal of the project work represented in `SESE_FIXED_THRESHOLDS (1).ipynb` is to implement SeSE in a way that can be executed and inspected end-to-end, and to explore a controlled “fixed-threshold” configuration while still allowing some refinement parameters to be optimized.

Key challenges addressed in the notebook implementation are:

- **Implementing SeSE as an explicit graph-and-entropy pipeline**: SeSE relies on constructing a directed semantic graph, computing structural entropy, and optimizing an encoding tree under a depth constraint. Each step has to be made concrete in code and kept numerically stable.
- **Supporting both short-form and claim-level (long-form) settings**: short-form operates on a response–response semantic graph; long-form operates on a claim–response bipartite graph.
- **Controlling which parameters are fixed vs. learnable**: in this notebook, parameters corresponding to the paper defaults are held fixed (e.g., merge threshold Delta = 0, depth discount lambda = 1, temperature tau = 1, contradiction weight gamma = 0), while additional refinement parameters remain learnable.
- **Making the implementation differentiable where training is intended**: the notebook includes explicit “bug fix” notes where earlier versions would have broken gradient flow to parameters.

## Brief recap of the proposed approach

The approach follows the paper’s main idea: treat the semantic space induced by multiple sampled outputs as a graph, and quantify the intrinsic uncertainty of that semantic space using *structural entropy* under an optimal hierarchical compression (an optimal encoding tree). A higher SeSE value indicates higher semantic uncertainty and therefore higher hallucination risk.

Two settings are used:

- **Short-form uncertainty**: given multiple sampled responses to the same prompt, construct a directed semantic graph over responses and compute SeSE for that response set.
- **Long-form (claim-level) uncertainty**: decompose a long response into atomic claims; connect sampled responses to claims via entailment (edges are 1 if a response entails a claim); compute claim-level SeSE along the encoding-tree path to each claim, then classify a claim as hallucination vs factual using a support threshold.

## System/model overview

### Paper-level system (SeSE)

SeSE is a black-box uncertainty quantification framework that does not require access to internal model activations. It uses semantic relations between multiple sampled outputs, rather than token probabilities, to estimate uncertainty.

For short-form generation, SeSE builds a **directed** semantic graph G_dir = (V, E, W) where vertices are sampled responses and edge weights encode directional semantic relationships derived from an NLI model. The graph is then sparsified adaptively to preserve structural information while removing noisy edges. SeSE is defined as the total structural entropy of an optimal encoding tree T* of the sparsified graph under a maximum depth constraint K.

For long-form generation, SeSE builds a **bipartite** claim–response graph G_cr = ((R, C), E) with binary entailment edges from responses to claims, then computes claim-level uncertainty using structural entropy along the encoding-tree path to the claim node.

### Notebook system (“fixed thresholds” implementation)

`SESE_FIXED_THRESHOLDS (1).ipynb` implements:

- `SeSEParams`: a parameter container with paper-default values and optional learnable refinement scalars. In this “fixed thresholds” notebook, several paper parameters are explicitly fixed (e.g., Delta = 0, lambda = 1, tau = 1, and gamma = 0).
- `SeSEShortForm`: takes an N×N matrix of pairwise NLI scores, builds a weighted graph, computes structural entropy, and returns a scalar SeSE score.
- `SeSELongForm`: takes an entailment matrix between R responses and C claims, builds a bipartite graph, computes an encoding tree, and returns per-claim SeSE scores plus a label based on support.

The notebook includes a training loop (`SeSETrainer`) that treats the SeSE score as a logit and uses `BCEWithLogitsLoss` to fine-tune parameters on labeled short-form examples. In the specific “fixed thresholds” configuration, only a subset of parameters are trainable.
### Enhanced project system: dual-path decoding architecture

The project extends the base SeSE framework with a **dual-path architecture** that systematically compares two complementary decoding and scaling strategies for uncertainty quantification:

**System pipeline architecture:**

1. **Input Stage:** Semantic responses from the LLM are fed into the pipeline.

2. **Temperature Scaling Stage σ(temp):** Input responses are normalized using temperature-adjusted scaling, controlling the confidence and concentration of the semantic space representation. Temperature acts as a hyperparameter controlling how "sharp" or "smooth" the uncertainty estimates become.

3. **Dual Decoding Branches:**
   - **Path A – Softmax Decoding (SySca):** Applies learned softmax-based normalization across semantic clusters, producing continuous probabilistic confidence scores. This path emphasizes smooth probability distributions and captures the full entropy of the semantic space.
   - **Path B – Fixed Scaling (SySla with λ parameter):** Applies fixed-threshold scaling to produce more discrete, threshold-driven decisions. This path emphasizes deterministic decision boundaries and robustness to small perturbations.

4. **Adaptive Decision Selection:** A decision module evaluates both paths' outputs and selects which decoding strategy or combination of strategies to prioritize based on:
   - Response confidence distribution characteristics
   - Entropy magnitude and structural complexity
   - Training signals from labeled data

5. **Multi-Stage Decoding:** Downstream decode operations apply claim-level or response-level classifiers to produce final hallucination/factuality decisions.

**Rationale for dual-path design:**
- **Probabilistic vs. deterministic trade-off:** Softmax decoding captures nuanced uncertainty gradients; fixed scaling provides interpretable, stable thresholds.
- **Hyperparameter exploration:** Allows systematic investigation of whether temperature scaling or fixed scaling parameters improve SeSE discrimination.
- **Hybrid robustness:** Comparing both approaches identifies which strategy generalizes better across different response types and hallucination patterns.
## Key technical components

### 1) Directed semantic graph construction (paper)

Given a prompt/context x and a sampled response set R = {r1, …, rN}, the paper measures directional semantic relations using an NLI model on ordered pairs (ri, rj). The NLI model outputs probabilities for entailment/neutral/contradiction:

- P_NLI(ri → rj | x) = [p_e, p_n, p_c] (Eq. 11 in the paper excerpt).

Edge weights are defined as a weighted combination of those probabilities:

- W(ri, rj) = A_ij = ω · P_NLI(ri → rj | x) with ω = (1, 1/2, 0)^T (Eq. 12 in the paper excerpt).

This yields a directed graph where A_ij need not equal A_ji.

The paper then **sparsifies** the initially dense graph using an adaptive k-NN selection that minimizes a one-dimensional structural entropy objective H1(G_k), selecting k* using the local-minimum condition shown as Eq. 13 in the excerpt.

### 2) Hierarchical abstraction via encoding-tree optimization (paper)

The paper defines the optimal encoding tree and SeSE as:

- T* = argmin over all trees T with height(T) ≤ K of H_T(G*_dir) (Eq. 14 in the excerpt).
- SeSE(G*_dir) = sum over nodes α in T* with α ≠ λ of H_{T*}(G*_dir; α) (Eq. 15 in the excerpt).

The tree is optimized greedily using “merging” and “combining” operators on sibling nodes. The entropy reduction for an operation is:

- DeltaSeSE_op(α, β) = H_T(G) − H_{T_{α,β}}(G*_dir) (Eq. 16 in the excerpt).

At each step, the operator that yields the largest positive entropy reduction is applied, stopping when no sibling pair yields a positive reduction or the maximum height K is reached.

### 3) Claim-level (long-form) graph and score (paper + notebook)

In long-form generation, the paper constructs a bipartite graph G_cr between sampled responses R and claims C, adding an edge when a response entails a claim (binary weights). Claim-level SeSE is defined as the cumulative entropy along the encoding-tree path from the root to the claim node; the excerpt includes:

- SeSE(G_cr; c) = − Σ … log2(…) (Eq. 17 shown in the excerpt).

In the notebook, claim classification uses a **support threshold**: a claim is labeled FACTUAL if at least a given fraction of sampled responses entail it; otherwise it is labeled HALLUCINATION.

## Experimental setup

This report describes the *executed* experimental outputs present in the notebook.

### Notebook configuration and demonstrations

- **Short-form demo**: two sets of 5 responses are constructed: a “low-uncertainty” set with consistent paraphrases, and a “high-uncertainty” set with conflicting alternatives.
- **Long-form demo**: 8 claims and 10 responses are used, producing an entailment matrix of shape 10×8. Claim labels are assigned using a fixed support threshold of **40%** (i.e., at least 4 of 10 responses must entail the claim to be considered factual).

### Metrics and outputs

The notebook demonstration outputs:

- A scalar SeSE score per response set (short-form).
- A per-claim table containing SeSE score, support percentage, support count, and a binary label (long-form).

The paper (separately) evaluates long-form detection using AUROC and AURAC; the excerpted Table II shows SeSE outperforming multiple baselines on two datasets (FActScore and PopQA) for two models (DeepSeek-V3.1 and Gemini-2.5-Flash). Those paper results are used here only as context for what “good performance” means in the original SeSE study; the notebook itself does not reproduce AUROC/AURAC for a full dataset.

## Results (main focus)

### 1) Short-form uncertainty scoring (notebook)

The notebook prints:

- Low-uncertainty SeSE score: **0.7973**
- High-uncertainty SeSE score: **0.8431**
- Difference (high − low): **+0.0458**

This is consistent with the intended behavior: the response set that contains conflicting candidate answers receives a higher SeSE score.

The notebook also reports a sensitivity sweep over alpha (entailment weight), showing that the separation between low and high uncertainty varies with alpha. In the printed sweep, the gap ranges from approximately **+0.0028** (near alpha = 0.37) to about **+0.0509** (near alpha = 0.64), indicating that edge-weighting choices can affect the discriminability of SeSE in this small demonstration.

### 2) Long-form claim-level hallucination classification (notebook)

Using support threshold **40%**, the notebook’s claim-level table contains 8 claims. The output labels **4/8** as FACTUAL and **4/8** as HALLUCINATION. Key entries (support and labels) are:

- Claim 0 (Everest highest): support **60%** (6/10) → **FACTUAL**, SeSE **0.1171**
- Claim 1 (Great Wall visible from space): support **20%** (2/10) → **HALLUCINATION**, SeSE **0.1893**
- Claim 4 (Python created by Guido): support **40%** (4/10) → **FACTUAL**, SeSE **0.1074**
- Claims 3, 5, 7 (Einstein invented lightbulb; 10% brain; Sun orbits Earth): support **0%** (0/10) → **HALLUCINATION**, SeSE **0.0884**

Two observations from this specific output are important:

1) **The classification decision in the notebook is driven by support, not by the SeSE score.** The label is assigned by comparing support fraction to the threshold, and SeSE is reported alongside as an uncertainty signal.
2) **SeSE values are not strictly ordered by the support fraction in this small demo.** For example, some 0%-support hallucinations have SeSE 0.0884, which is lower than some factual claims. This is not automatically a failure of SeSE; it indicates that (i) this demo uses a small, hand-constructed entailment pattern, and (ii) structural entropy depends on global graph structure and path/community relationships, not only on one node’s degree or support count.

### 3) Context from the paper’s long-form results (notebook-aligned objective, paper-level evaluation)

The paper excerpted Table II reports that, on long-form hallucination detection, SeSE achieves higher AUROC and AURAC than a set of baselines (including DSE, verbalized confidence methods, and centrality metrics). For example, for Gemini-2.5-Flash on PopQA, SeSE reports **AUROC 0.8859** and **AURAC 0.8216** (Table II excerpt). These values illustrate the paper’s claim that structural information improves discrimination in realistic long-form settings; they are not computed by the notebook demo.

## Justification and reasoning behind results

### Why the short-form scores behave as expected

In short-form, the low-uncertainty response set consists of responses that largely entail one another or are semantically compatible. That yields a graph with more coherent directional relationships and, after sparsification and hierarchical abstraction, a lower structural entropy. The high-uncertainty set introduces conflicting candidate answers, which weakens coherent entailment structure and increases uncertainty in random walks on the semantic graph, raising the structural entropy score.

The alpha sensitivity sweep supports this interpretation: alpha controls how strongly entailment edges dominate the weight matrix. When entailment is under-weighted or over-weighted relative to the observed NLI pattern, the structural signal that separates “coherent” vs “disordered” sets can become smaller.

### Why claim labels can disagree with simple SeSE ordering in the demo table

In the notebook, claim factuality is decided by a fixed rule: support fraction ≥ 0.4 is FACTUAL. SeSE is computed as a path entropy on the encoding tree built from the full bipartite graph. A claim can have low support but still sit in a portion of the encoding tree whose structural contribution is small (for instance, if several unsupported claims share similar interaction patterns and are grouped early). Conversely, a supported claim could be placed in a higher-entropy region if it interacts with a structurally ambiguous subset of responses.

This is consistent with the paper’s conceptual meaning: SeSE is intended to capture uncertainty as a *structural* property of semantic interactions, not merely as “how many responses agree.”

## Implications and insights

- **SeSE provides a structural uncertainty score that can separate coherent vs conflicting response sets**, as shown by the short-form demo’s higher score for the disordered set.
- **In claim-level settings, support-based labeling provides a clear operational definition of factuality**, but SeSE provides a complementary signal: it indicates which claims lie in structurally uncertain regions of the claim–response interaction graph.
- **Parameter choices matter**, even in a small demo: the alpha sweep shows the separation between low/high uncertainty depends on edge weighting. This supports treating the weighting and sparsification settings as part of the experimental configuration rather than as incidental details.

## Limitations

- **Notebook results are demonstrations, not a benchmark reproduction.** The long-form output is a small, hand-constructed example (8 claims, 10 responses) and does not compute AUROC/AURAC.
- **Labeling is rule-based in the notebook.** Claims are labeled by support threshold, which is not the same as evaluating SeSE as a classifier with a tuned decision threshold.
- **Entailment estimation is simplified in the demo.** The notebook uses an entailment matrix for the long-form example rather than running a full NLI/LLM entailment judge over real outputs in the recorded run.
- **SeSE values depend on global structure.** Interpreting a single claim’s SeSE score requires considering its position in the encoding tree and the interaction structure of the entire bipartite graph.

## Conclusion

This work implements the SeSE framework and demonstrates its intended behavior in two settings. In short-form uncertainty quantification, SeSE assigns a higher structural-entropy score to a response set containing conflicting candidates (0.8431) than to a coherent paraphrase set (0.7973), producing a positive separation of +0.0458. In long-form claim-level analysis, the notebook constructs a claim–response bipartite interaction graph, reports per-claim SeSE scores, and classifies claims using a fixed 40% support threshold, yielding 4 factual and 4 hallucinated claims in the demo. The paper’s experimental results (reported separately) show that SeSE improves AUROC and AURAC over multiple baselines in long-form settings, supporting the broader motivation for structural-information-guided uncertainty quantification.

