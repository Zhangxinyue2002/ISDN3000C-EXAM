# ISDN3000C Exam Cheatsheet: Lecture 6 (AI White-Box) + Lecture 7 (Neural Networks)

Assumptions: Slides aren’t text-extractable from PDFs, so this guide covers the most likely syllabus for “AI White-Box” and “Neural Networks.” It’s optimized for open-book quick search during the exam.

---

## Quick search index (A–Z)

- Activation functions → [Neural nets > Activations](#activation-functions)
- Adam/optimizers → [Neural nets > Optimization](#optimization-algorithms-and-schedules)
- Backpropagation → [Neural nets > Backprop](#backpropagation-essentials)
- Batch normalization → [Regularization/generalization](#regularization-and-generalization)
- Chain rule → [Backprop](#backpropagation-essentials)
- CNNs → [Convolutional nets](#cnn-basics)
- Confusion matrix/Precision/Recall/F1 → [Metrics](#classification-metrics-and-curves)
- Counterfactual explanations → [XAI methods](#explainability-xai-for-black-box-models)
- Decision trees → [White-box > Trees](#decision-trees-id3c45-basics)
- Entropy/Information Gain → [White-box > Trees](#entropy-and-information-gain)
- Fairness/ethics → [White-box > Responsible AI](#responsible-ai-fairness-transparency)
- Feature importance (Permutation/SHAP) → [XAI methods](#explainability-xai-for-black-box-models)
- Forward vs backward chaining → [Rule-based systems](#rule-based-and-expert-systems)
- Gradients/vanishing/exploding → [NN training issues](#vanishingexploding-gradients)
- Initialization → [Training tips](#practical-training-tips)
- L1/L2/Weight decay → [Regularization/generalization](#regularization-and-generalization)
- LIME vs SHAP → [XAI methods](#explainability-xai-for-black-box-models)
- Logistic regression interpretability → [White-box models](#white-box-ml-models-linear--logistic)
- Loss functions → [Losses](#loss-functions-quick-view)
- MSE (mean squared error) → [Lecture 6 > Regression](#regression-basics-mse)
- Gradient Descent (learning rate) → [Lecture 6 > Optimization](#gradient-descent)
- Logistic Regression (sigmoid) → [Lecture 6 > Classification](#logistic-regression)
- MLP/perceptron → [Neural nets basics](#neural-net-basics-perceptron-and-mlp)
- Optimization (SGD, Momentum, RMSProp, Adam) → [Optimization](#optimization-algorithms-and-schedules)
- Overfitting/underfitting → [Regularization/generalization](#regularization-and-generalization)
- PDP/ICE → [XAI methods](#explainability-xai-for-black-box-models)
- ROC/PR curves → [Metrics](#classification-metrics-and-curves)
- RNN/LSTM/GRU → [Sequence models](#rnnlstmgru-basics)
- SHAP basics/formula intuition → [XAI methods](#explainability-xai-for-black-box-models)
- Softmax + Cross-entropy gradient → [Formulas](#formula-quick-reference)
- Perceptron/XOR/Activation (ReLU) → [Lecture 7 > Basics](#xor-and-activations)
- Convolution/filters/kernels → [Lecture 7 > CNN](#cnn-convolution-basics)

---

## Lecture summaries (grounded in slides)

### Lecture 6: AI White-Box (from slides)

#### AI basics and framing
- Adaptive systems that take rational actions based on observations; agents choose actions to maximize expected utility.
- Framing prediction tasks: define features (inputs), model f(.), prediction ŷ, and target/label. Example: machine failure prediction using vibration, age, pressure, temperature.

#### Regression basics (MSE)
- Linear model: y = m x + b (extendable to multiple features y = w·x + b).
- Loss: Mean Squared Error (MSE) averages squared residuals to penalize large errors and avoid cancellation.

#### Gradient Descent
- Iterative optimization: step opposite the gradient to reduce loss; key hyperparameter is learning rate α (too big overshoots; too small is slow).
- Extends naturally to multiple features (higher-dimensional parameter space).

#### Classification with white-box models
- Sigmoid maps scores to [0,1]; Logistic Regression = linear score + sigmoid for probabilistic classification.
- Decision Trees learn IF/ELSE splits greedily to maximize purity of child nodes.
  - Impurity metrics: Entropy, Gini; Information Gain = impurity(parent) − weighted impurities(children).
  - Training: for each feature and split point, evaluate resulting purity; choose best split; recurse; pros: interpretable; cons: can overfit/high variance.

Anchors: [Regression](#regression-basics-mse) • [Gradient Descent](#gradient-descent) • [Logistic Regression](#logistic-regression) • [Decision Trees](#decision-trees-id3c45-basics)

### Lecture 7: Neural Networks (from slides)

#### Motivation and feature learning
- Structured vs unstructured data: explicit features vs implicit/complex features (images, audio, text).
- From manual feature engineering to feature learning: models learn useful representations directly from raw data.

#### Perceptron, XOR, and activations
- Perceptron is a linear classifier: z = w·x + b; only solves linearly separable problems.
- XOR requires non-linear decision boundaries → introduce non-linear activations (e.g., ReLU) and multiple layers.

#### MLP structure and training
- Layers: input → hidden (learn intermediate representations) → output.
- Universal Approximation: with at least one hidden layer + nonlinearity, MLP can approximate any continuous function.
- Training loop: forward pass → loss → backpropagation (compute gradients) → optimizer step (update weights).

#### CNN convolution basics
- Motivation: MLPs on images destroy spatial structure and explode parameters; CNNs leverage locality and weight sharing.
- Convolution: slide small kernels/filters over the image to produce feature maps; filters learn to detect edges, textures, etc.

Anchors: [Perceptron/XOR](#xor-and-activations) • [MLP](#neural-net-basics-perceptron-and-mlp) • [Backprop](#backpropagation-essentials) • [CNN](#cnn-basics)

## Lecture 6: AI White-Box

### What “white-box” vs “black-box” means
- White-box: Model structure is directly interpretable (rules, trees, linear coefficients). Easy to audit/explain.
- Black-box: High predictive performance but opaque (deep nets, boosted ensembles). Requires post-hoc explainability.
- Trade-off: Interpretability vs accuracy vs complexity; domain/regulation may mandate interpretability.

### Rule-based and Expert Systems
- Components: Knowledge base (facts + rules), inference engine, working memory, explanation facility.
- Rule form: IF conditions (antecedent) THEN action/conclusion (consequent). Often Horn clauses.
- Inference:
  - Forward chaining (data-driven): Start from facts, apply rules to derive new facts until goal.
  - Backward chaining (goal-driven): Start from query/goal, prove by finding rules that conclude the goal and recursively proving their premises.
- Pros: Transparent reasoning, easy to justify. Cons: Knowledge acquisition bottleneck, brittle, poor with uncertainty/noise.

### Decision Trees (ID3/C4.5 basics)
- Greedy top-down partitioning of feature space by splits that increase “purity.”
- Splitting criteria:
  - ID3: Information Gain (entropy decrease).
  - C4.5: Gain Ratio (adjusts for intrinsic information), handles continuous features, missing values.
  - CART: Gini impurity for classification; MSE for regression.
- Stopping/pruning: Min samples per leaf, max depth, post-pruning to avoid overfit.
- Interpretability: Path from root→leaf is a human-readable rule.

#### Entropy and Information Gain
- Entropy: H(Y) = − Σ p(y) log2 p(y)
- Conditional Entropy: H(Y|X) = Σ p(x) H(Y|X=x)
- Information Gain: IG(Y, X) = H(Y) − H(Y|X)
- Gain Ratio (C4.5): IG / IntrinsicInfo(X) where IntrinsicInfo(X) = − Σ p(x) log2 p(x)

### White-box ML models (Linear / Logistic)
- Linear regression: y ≈ w·x + b; coefficients quantify direction/magnitude.
- Logistic regression: p(y=1|x) = σ(w·x + b); odds ratio interpretable per feature.
- Pros: Simple, global interpretability; Cons: limited expressiveness, sensitive to multicollinearity.

### Explainability (XAI) for black-box models
- Global vs local explanations:
  - Global: Overall feature effects (permutation importance, PDP/ICE, global SHAP).
  - Local: Instance-level attribution (LIME, SHAP, Counterfactuals).
- Methods:
  - Permutation importance: Drop in performance when shuffling a feature.
  - PDP: Average model prediction when varying one feature.
  - ICE: PDP per-instance curves (heterogeneity visible).
  - LIME: Local linear surrogate; sample near instance, fit weighted linear model.
  - SHAP: Game-theoretic Shapley values; consistent additive feature attributions.
  - Counterfactuals: Minimal changes to features to flip the prediction.
- Caveats: Correlated features can mislead; distribution shift invalidates explanations; ensure faithfulness and stability.

### Classification metrics and curves
- Confusion matrix: TP, FP, TN, FN.
- Accuracy = (TP+TN)/(All) — misleading on imbalanced data.
- Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R).
- ROC-AUC: rank-based; PR-AUC more informative with class imbalance.

### Responsible AI (Fairness, Transparency)
- Fairness definitions: demographic parity, equal opportunity, equalized odds.
- Bias sources: data, labels, proxies; mitigation: reweighing, constraints, post-processing.
- Documentation: model cards, data statements; governance and audit trails.

---

## Lecture 7: Neural Networks

### Neural net basics: Perceptron and MLP
- Perceptron: linear threshold unit; not linearly separable problems (e.g., XOR) need hidden layers.
- MLP: layers of linear transformations + nonlinear activation.
- Forward pass (one hidden layer):
  - h = φ(W1 x + b1)
  - ŷ = f(W2 h + b2) (f = softmax for multi-class, sigmoid for binary)

### Activation functions
- Sigmoid: σ(z) = 1/(1+e^(−z)); derivative σ(1−σ); saturates.
- Tanh: zero-centered; derivative 1−tanh^2.
- ReLU: max(0, z); non-saturating; dead ReLUs possible.
- LeakyReLU/ELU/GELU: mitigate dead units, improve gradients.

### Loss functions (quick view)
- Regression: MSE, MAE, Huber.
- Binary classification: BCE (log loss).
- Multi-class: Cross-entropy with softmax.

### Backpropagation essentials
- Uses chain rule through computational graph to compute ∂L/∂θ.
- For linear layer y = Wx + b with upstream gradient g = ∂L/∂y:
  - ∂L/∂W = g x^T; ∂L/∂x = W^T g; ∂L/∂b = g (sum over batch).
- Softmax + CE simplifies to p − y (for one-hot y) at output.

### Optimization algorithms and schedules
- SGD: θ ← θ − η ∇L
- Momentum: v ← βv + (1−β)∇L; θ ← θ − η v
- RMSProp: adaptive per-parameter step via EMA of squared grads.
- Adam: m,v EMAs with bias correction; popular default.
- Schedules: step/plateau decay, cosine, warmup; tune learning rate carefully.

### Regularization and generalization
- L2 (weight decay): adds λ||w||^2; shrinks weights.
- L1: sparsity; feature selection.
- Dropout: randomly zero activations; at test-time scale.
- Early stopping: monitor val metric to stop before overfit.
- BatchNorm: stabilize/accelerate training, mild regularizer.
- Data augmentation: flips/crops/noise; mixup/cutmix (vision).

### CNN basics
- Convolution: local receptive fields and shared weights.
- Output size (1D): out = floor((n + 2p − k)/s) + 1; Params = k*k*C_in*C_out + C_out (bias)
- Pooling: downsample (max/avg) to gain invariances.

### RNN/LSTM/GRU basics
- RNN: h_t = φ(W_x x_t + W_h h_{t−1} + b)
- Problems: vanishing/exploding gradients on long sequences.
- LSTM/GRU: gating mechanisms to preserve/forget information.
- Mitigations: gating, residuals, layer norm, gradient clipping, shorter BPTT windows.

### Vanishing/exploding gradients
- Causes: repeated multiplication by |<1| or |>1| Jacobians; saturating activations; deep unnormalized stacks.
- Fixes: ReLU/GELU, proper init (He/Xavier), normalization, residuals, gating, clipping.

### Practical training tips
- Standardize inputs; use correct output/activation/loss pair.
- Start with Adam, LR 1e−3 (classification), then tune; use LR finder or small grid.
- Verify shapes, monitor gradients, overfit a tiny batch to sanity-check.
- Seed results; record versions; use train/val/test splits properly.

---

## Predicted exam questions with model answers

1) Explain forward vs backward chaining; give a small example.
- Forward: From facts F={A,B}, rules R1: A∧B→C, R2: C→D → derive C then D.
- Backward: To prove D, need C (R2), to prove C need A,B (R1); check facts.
- When: Forward for data streaming/diagnosis; backward for query answering/expert consultation.

2) Compute entropy and information gain for a split.
- Suppose Y has p+=0.6, p−=0.4: H(Y)=−0.6log2 0.6 −0.4log2 0.4≈0.971.
- Feature X creates subsets with entropies 0.811 and 0.0 with weights 2/3 and 1/3 → H(Y|X)=0.541.
- IG(Y,X)=0.971−0.541=0.430.

3) Why are decision trees considered white-box? Pros/cons vs neural nets.
- Paths are readable rules; easy to justify. Pros: transparency, handling of mixed types, little prep. Cons: high variance, axis-aligned splits, less accurate than ensembles/NNs on complex patterns.

4) LIME vs SHAP differences and when to use.
- LIME: local linear surrogate; faster but less consistent; good for quick per-instance explanations.
- SHAP: Shapley-consistent attributions; more faithful but costlier; good for auditing/feature contribution.

5) Derive the gradient for a 1-hidden-layer MLP with softmax + CE.
- Output gradient δ_out = p − y.
- W2 grad: δ_out h^T; b2 grad: δ_out (sum over batch).
- Hidden gradient: δ_h = (W2^T δ_out) ⊙ φ′(z1); then W1 grad: δ_h x^T; b1 grad: δ_h.

6) Explain vanishing gradients and two mitigations.
- Repeated multiplications shrink gradients; use ReLU/GELU, residual connections, good init, normalization, or GRU/LSTM for sequences.

7) Given a confusion matrix, compute Precision, Recall, F1.
- P=TP/(TP+FP); R=TP/(TP+FN); F1=2PR/(P+R). Include a quick numeric example if asked.

8) Compare L2 regularization and dropout.
- L2: shrinks weights uniformly; keeps all features; smooths loss landscape.
- Dropout: randomly drops activations; acts like model averaging; stronger regularizer.

9) For a convolution layer, compute output size and parameter count.
- Example: input 32×32×3, k=3, p=1, s=1, C_out=64 → out 32×32×64; params = 3×3×3×64 + 64 = 1,792.

10) Why batch normalization helps training.
- Stabilizes activation distributions, allows higher LR, smooths optimization; also mild regularization.

11) Counterfactual explanation concept and constraints.
- Minimal feature perturbation to flip label; should be plausible, actionable, and sparse.

12) Backward vs forward selection for tree pruning (high-level).
- Pre-pruning (forward): stop conditions early. Post-pruning (backward): grow full tree, then cut subtrees by validation or cost-complexity.

---

## Formula quick reference

- Entropy: H(Y) = − Σ p(y) log2 p(y)
- Conditional: H(Y|X) = Σ_x p(x) H(Y|X=x)
- Information Gain: IG = H(Y) − H(Y|X)
- Gain Ratio: IG / (−Σ p(x) log2 p(x))
- Sigmoid: σ(z)=1/(1+e^(−z)); σ′=σ(1−σ)
- Tanh: tanh′=1−tanh^2
- ReLU: max(0,z)
- Softmax: p_i = exp(z_i)/Σ_j exp(z_j)
- CE grad (softmax): ∂L/∂z = p − y
- Linear layer: ∂L/∂W = g x^T; ∂L/∂x = W^T g; ∂L/∂b = g
- L2 regularization: add λ||w||^2 → ∂L/∂w += 2λw (or λw with weight decay convention)
- Convolution output (1D): out=floor((n+2p−k)/s)+1; Params=k*k*C_in*C_out + C_out

---

## Glossary (fast lookup)
- BCE: Binary Cross-Entropy. CE: Cross-Entropy. AUC: Area under ROC/PR curve.
- PDP/ICE: Partial dependence / Individual conditional expectation.
- LIME/SHAP: Local interpretable model-agnostic explanations / Shapley additive explanations.
- BPTT: Backpropagation through time.
- BN: BatchNorm. WD: Weight Decay.

---

## How to use during exam
- Use the Quick search index and Glossary to jump. Search this file by keyword (Ctrl+F).
- For math, jump to Formula quick reference.
- For conceptual “compare/contrast,” check Predicted exam Q&A.

---

If you can share any specific slide topics or emphasis, I can extend this sheet with tailored examples and derivations.
