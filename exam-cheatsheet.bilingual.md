# ISDN3000C Exam Cheatsheet / 开卷考试速查表 (EN + 中文)

Use the links below to jump to your preferred language. Both versions include quick-search indexes and the same content structure.

- English version → [Jump](#english-version)
- 中文版本 → [跳转](#中文版本)

---

## English quick search index (A–Z)

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

## 中文快速索引（A–Z）

- 激活函数 → [神经网络 > 激活函数](#activation-functions)
- Adam/优化器 → [神经网络 > 优化算法](#optimization-algorithms-and-schedules)
- 反向传播 → [神经网络 > 反向传播](#backpropagation-essentials)
- 批量归一化（BatchNorm）→ [正则化/泛化](#regularization-and-generalization)
- 链式法则 → [反向传播](#backpropagation-essentials)
- 卷积神经网络（CNN）→ [卷积基础](#cnn-basics)
- 混淆矩阵/精确率/召回率/F1 → [分类指标](#classification-metrics-and-curves)
- 反事实解释 → [可解释性 XAI](#explainability-xai-for-black-box-models)
- 决策树 → [白盒 > 树模型](#decision-trees-id3c45-basics)
- 熵/信息增益 → [白盒 > 树模型](#entropy-and-information-gain)
- 公平性/伦理 → [白盒 > 负责任的 AI](#responsible-ai-fairness-transparency)
- 特征重要性（置换/SHAP）→ [可解释性 XAI](#explainability-xai-for-black-box-models)
- 正向/反向链推 → [规则/专家系统](#rule-based-and-expert-systems)
- 梯度消失/爆炸 → [NN 训练问题](#vanishingexploding-gradients)
- 初始化 → [训练技巧](#practical-training-tips)
- L1/L2/权重衰减 → [正则化/泛化](#regularization-and-generalization)
- LIME vs SHAP → [可解释性 XAI](#explainability-xai-for-black-box-models)
- 逻辑回归可解释性 → [白盒模型](#white-box-ml-models-linear--logistic)
- 损失函数 → [损失函数总览](#loss-functions-quick-view)
- MSE（均方误差）→ [讲义 6 > 回归](#regression-basics-mse)
- 梯度下降（学习率）→ [讲义 6 > 优化](#gradient-descent)
- 逻辑回归（Sigmoid）→ [讲义 6 > 分类](#logistic-regression)
- 感知器/MLP → [神经网络基础](#neural-net-basics-perceptron-and-mlp)
- 优化（SGD/Momentum/RMSProp/Adam）→ [优化算法](#optimization-algorithms-and-schedules)
- 过拟合/欠拟合 → [正则化/泛化](#regularization-and-generalization)
- PDP/ICE → [可解释性 XAI](#explainability-xai-for-black-box-models)
- ROC/PR 曲线 → [指标](#classification-metrics-and-curves)
- RNN/LSTM/GRU → [序列模型](#rnnlstmgru-basics)
- SHAP 基本概念 → [可解释性 XAI](#explainability-xai-for-black-box-models)
- Softmax + 交叉熵梯度 → [公式](#formula-quick-reference)
- 感知器/XOR/激活（ReLU）→ [讲义 7 > 基础](#xor-and-activations)
- 卷积/滤波器/核 → [讲义 7 > CNN](#cnn-convolution-basics)

---

## English version

[The full English cheatsheet content is included below. It mirrors your `exam-cheatsheet.md` for one-file convenience.]

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

Easy idea: “See → Think → Act.” We observe numbers (features), use a recipe (model) to guess an answer (prediction), and compare to the truth (label).
Remember 3 things:
- Input = features you measure. Output = what you want to know.
- A model is just a formula or flow of simple steps.
- We judge a model by how small its errors are on known answers.

#### Regression basics (MSE)
- Linear model: y = m x + b (extendable to multiple features y = w·x + b).
- Loss: Mean Squared Error (MSE) averages squared residuals to penalize large errors and avoid cancellation.

In plain words:
- Draw a straight line that best fits the dots. The “best” line makes squared vertical distances tiny.

Mini example:
- True y: [2, 3]; Pred y: [1, 4] → errors: [+1, −1]; squares: [1, 1]; MSE = (1+1)/2 = 1.

Common mistakes:
- Mixing up x and y (input vs prediction). Use ŷ = w·x + b.
- Forgetting to square errors (then + and − cancel out).

Beginner mode: Regression & MSE — super simple
- Goal: Draw a line that best guesses y from x.
- Error for one point = actual − predicted. Squared makes big errors hurt more.
- MSE = average of all squared errors. Smaller is better.

Try it yourself (1 minute):
- Data: (x,y) = (1,2), (2,3). Suppose line is ŷ = x + 0.
- Predictions = [1,2] → errors = [1,1] → squares = [1,1] → MSE = 1.
- If you move the line up by +1: ŷ = x + 1 → preds [2,3] → errors [0,0] → MSE = 0 (perfect for this toy set).

Pitfalls to avoid:
- Units matter. If x is in meters and y in dollars, rescale features if needed.
- Outliers can dominate MSE. Consider robust losses (MAE/Huber) if needed.

#### Gradient Descent
- Iterative optimization: step opposite the gradient to reduce loss; key hyperparameter is learning rate α (too big overshoots; too small is slow).
- Extends naturally to multiple features (higher-dimensional parameter space).

Analogy: You’re on a foggy hill. You feel the steepest downward direction and take a small step. Repeat until flat.
Quick rules:
- If loss jumps up and down wildly → step size is too big (reduce α).
- If loss crawls slowly → α too small (increase a bit).

Beginner mode: Gradient Descent — super simple
- Imagine rolling a ball downhill on the loss surface.
- Learning rate = step size. Too big: you bounce around; too small: you crawl.
- Stop when steps don’t reduce loss anymore (or after a patience setting).

Tiny demo thought experiment:
- Start loss = 10. Take a step → loss 6. Another step → loss 4. Reduce α if the loss ever goes up; increase a bit if the loss barely changes for many steps.

#### Classification with white-box models
- Sigmoid maps scores to [0,1]; Logistic Regression = linear score + sigmoid for probabilistic classification.
- Decision Trees learn IF/ELSE splits greedily to maximize purity of child nodes.
  - Impurity metrics: Entropy, Gini; Information Gain = impurity(parent) − weighted impurities(children).
  - Training: for each feature and split point, evaluate resulting purity; choose best split; recurse; pros: interpretable; cons: can overfit/high variance.

Explain like I’m new:
- Logistic regression: add up weighted features → squeeze with sigmoid to get a probability between 0 and 1.
- Decision tree: it asks yes/no questions (like 20 Questions) to split the data until each group is mostly one label.

Tiny IG example:
- Parent impurity = 0.8; after split, children impurities = 0.5 (weight 0.6) and 0.2 (weight 0.4).
- Weighted child = 0.6×0.5 + 0.4×0.2 = 0.3 + 0.08 = 0.38; IG = 0.8 − 0.38 = 0.42 (good).

Beginner mode: Logistic Regression — super simple
- Think: score = w·x + b (a weighted sum). Probability = sigmoid(score) which squeezes any number into 0..1.
- Decision: if p ≥ threshold (often 0.5), predict positive; else negative.
- Tip: shift threshold to trade precision vs recall.

Beginner mode: Decision Tree — super simple
- Ask the question that best splits the data into purer groups (highest IG).
- Repeat on each group until groups are pure enough or you hit stop rules.
- Prune back if the tree gets too specific (overfit).

Anchors: [Regression](#regression-basics-mse) • [Gradient Descent](#gradient-descent) • [Logistic Regression](#logistic-regression) • [Decision Trees](#decision-trees-id3c45-basics)

### Lecture 7: Neural Networks (from slides)

#### Motivation and feature learning
- Structured vs unstructured data: explicit features vs implicit/complex features (images, audio, text).
- From manual feature engineering to feature learning: models learn useful representations directly from raw data.

#### Perceptron, XOR, and activations
- Perceptron is a linear classifier: z = w·x + b; only solves linearly separable problems.
- XOR requires non-linear decision boundaries → introduce non-linear activations (e.g., ReLU) and multiple layers.

Quick intuition:
- One straight line can’t separate an “XOR” pattern. Stack layers + use bends (nonlinear activations) so the boundary can curve.

#### MLP structure and training
- Layers: input → hidden (learn intermediate representations) → output.
- Universal Approximation: with at least one hidden layer + nonlinearity, MLP can approximate any continuous function.
- Training loop: forward pass → loss → backpropagation (compute gradients) → optimizer step (update weights).

Backprop (no calculus needed):
- Ask “who caused the error?” and “by how much?”. Pull error backwards layer by layer, and nudge each weight in the helpful direction.

Sanity check ritual:
- Overfit a tiny batch (like 8 samples). If loss can’t go near zero, there’s a bug (shapes, LR, activation/loss mismatch).

#### CNN convolution basics
- Motivation: MLPs on images destroy spatial structure and explode parameters; CNNs leverage locality and weight sharing.
- Convolution: slide small kernels/filters over the image to produce feature maps; filters learn to detect edges, textures, etc.

Tiny 3×3 filter example:
- Input patch (3×3) · Filter (3×3) → multiply-and-sum 9 numbers → one output pixel.
- Slide the filter by 1 pixel → repeat. One filter = one feature map.

Why it helps:
- Same filter (shared weights) finds the same pattern anywhere → fewer parameters, stronger generalization.

ASCII slide-and-sum (2×2 filter on 4×4 input, stride 1):

```
Input (4x4)                Filter (2x2)
+--+--+--+--+              +--+--+
| 1| 2| 3| 4|              | a| b|
+--+--+--+--+              +--+--+
| 5| 6| 7| 8|      with    | c| d|
+--+--+--+--+              +--+--+
| 9|10|11|12|
+--+--+--+--+
|13|14|15|16|
+--+--+--+--+

First output (top-left): 1*a + 2*b + 5*c + 6*d
Slide right by 1 → use (2,3,6,7) → next output, etc.
```

Beginner mode: CNN — super simple
- Stride = how far the filter jumps each move (bigger stride → smaller output).
- Padding = add zeros around the image edges so size doesn’t shrink too fast.
- Pooling = summarize nearby values (max/avg) to make features more stable to small shifts.
- Param count for conv: kernel_h×kernel_w×Cin×Cout + Cout (biases).

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

If you only remember 3 things:
1) Trees pick splits that make groups purer.
2) Too deep trees memorize noise → prune or limit depth.
3) Path = IF…THEN… rule you can explain to a non-technical person.

ASCII sketch (tiny example):

```
            +-----------+
            | Outlook?  |
            +-----------+
              /       \
           Sunny      Rain
            /           \
     +-----------+    +----------+
     | Humidity? |    | Windy?   |
     +-----------+    +----------+
       /      \         /     \
    High     Low     False    True
     No       Yes      Yes      No
```

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

Quick chooser:
- Want a fast global feel? → Permutation importance, PDP.
- Per-instance “why this prediction?” → LIME (fast), SHAP (more faithful but slower).
- Need actionable change? → Counterfactuals (smallest tweak to flip the result).

### Classification metrics and curves
- Confusion matrix: TP, FP, TN, FN.
- Accuracy = (TP+TN)/(All) — misleading on imbalanced data.
- Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R).
- ROC-AUC: rank-based; PR-AUC more informative with class imbalance.

1-minute metrics map:
- Accuracy lies when one class dominates (e.g., 99% negatives).
- Precision asks: “Of my positives, how many were correct?”
- Recall asks: “Of all true positives, how many did I find?”
- F1 balances precision and recall when you need one number.

ASCII confusion matrix (positive = 1, negative = 0):
```
               Pred 1    Pred 0
Actual 1        TP        FN
Actual 0        FP        TN
```

Beginner mode: Precision, Recall, F1 — super simple
- Precision = when I say “positive,” how often am I right? (TP/(TP+FP))
- Recall = of all the real positives, how many did I catch? (TP/(TP+FN))
- F1 = a single score that balances both. Good when classes are imbalanced.
- Threshold trick: move threshold right → higher precision, lower recall; left → lower precision, higher recall.

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

When to pick:
- Try ReLU first. If many dead neurons or unstable grads, try LeakyReLU or GELU.
- For RNNs, tanh/sigmoid often appear inside gates (not for deep stacks).

### Loss functions (quick view)
- Regression: MSE, MAE, Huber.
- Binary classification: BCE (log loss).
- Multi-class: Cross-entropy with softmax.

### Backpropagation essentials
- Uses chain rule through computational graph to compute ∂L/∂θ.
- For linear layer y = Wx + b with upstream gradient g = ∂L/∂y:
  - ∂L/∂W = g x^T; ∂L/∂x = W^T g; ∂L/∂b = g (sum over batch).
- Softmax + CE simplifies to p − y (for one-hot y) at output.

Intuition for p − y:
- If the model assigns 0.9 to the true class (y=1), gradient = 0.9−1 = −0.1 → “lower the score a bit? no, raise others less.”
- If it assigns 0.1 to the true class, gradient = 0.1−1 = −0.9 → “big push to raise the true class score.”

Beginner mode: Backprop — super simple
- Backprop answers: which weights caused the mistake, and by how much?
- Linear layer y = W x + b, loss L: upstream gradient g = ∂L/∂y tells how sensitive loss is to y.
- Then: ∂L/∂W = g x^T (weights connecting inputs to outputs get credit proportional to input and error), ∂L/∂b = g, ∂L/∂x = W^T g.
- Intuition: if an input feature is big and the error is big, the connecting weight gets a bigger nudge.

### Optimization algorithms and schedules
- SGD: θ ← θ − η ∇L
- Momentum: v ← βv + (1−β)∇L; θ ← θ − η v
- RMSProp: adaptive per-parameter step via EMA of squared grads.
- Adam: m,v EMAs with bias correction; popular default.
- Schedules: step/plateau decay, cosine, warmup; tune learning rate carefully.

Quick starter pack:
- Adam with LR 1e−3 (classification) or 1e−4~1e−3 (vision) is a safe first try.
- If loss plateaus, reduce LR on plateau or try cosine decay with warmup.

### Regularization and generalization
- L2 (weight decay): adds λ||w||^2; shrinks weights.
- L1: sparsity; feature selection.
- Dropout: randomly zero activations; at test-time scale.
- Early stopping: monitor val metric to stop before overfit.
- BatchNorm: stabilize/accelerate training, mild regularizer.
- Data augmentation: flips/crops/noise; mixup/cutmix (vision).

Common overfitting smell tests:
- Training loss ↓, validation loss ↑ → you’re overfitting.
- Fixes: more data, stronger augmentation, more regularization (L2/Dropout), earlier stop, simpler model.

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

Tip: Memorize “parent minus weighted children.” If IG is big, the question (split) is useful.

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

Quick numeric example:
- TP=8, FP=2, FN=4 → P=8/10=0.8, R=8/12≈0.667, F1≈2*0.8*0.667/(0.8+0.667)≈0.727.

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

## 中文版本

[下面为完整中文速查表，内容与 `exam-cheatsheet.zh-CN.md` 一致，汇总于同一文件便于使用。]

# ISDN3000C 开卷考试速查表：讲义 6（AI 白盒）+ 讲义 7（神经网络）

说明：本中文版依据你的讲义内容整理，保留了内部锚点，便于在考试中快速 Ctrl+F 定位关键词。

---

## 快速索引（A–Z）

- 激活函数 → [神经网络 > 激活函数](#activation-functions)
- Adam/优化器 → [神经网络 > 优化算法](#optimization-algorithms-and-schedules)
- 反向传播 → [神经网络 > 反向传播](#backpropagation-essentials)
- 批量归一化（BatchNorm）→ [正则化/泛化](#regularization-and-generalization)
- 链式法则 → [反向传播](#backpropagation-essentials)
- 卷积神经网络（CNN）→ [卷积基础](#cnn-basics)
- 混淆矩阵/精确率/召回率/F1 → [分类指标](#classification-metrics-and-curves)
- 反事实解释 → [可解释性 XAI](#explainability-xai-for-black-box-models)
- 决策树 → [白盒 > 树模型](#decision-trees-id3c45-basics)
- 熵/信息增益 → [白盒 > 树模型](#entropy-and-information-gain)
- 公平性/伦理 → [白盒 > 负责任的 AI](#responsible-ai-fairness-transparency)
- 特征重要性（置换/SHAP）→ [可解释性 XAI](#explainability-xai-for-black-box-models)
- 正向/反向链推 → [规则/专家系统](#rule-based-and-expert-systems)
- 梯度消失/爆炸 → [NN 训练问题](#vanishingexploding-gradients)
- 初始化 → [训练技巧](#practical-training-tips)
- L1/L2/权重衰减 → [正则化/泛化](#regularization-and-generalization)
- LIME vs SHAP → [可解释性 XAI](#explainability-xai-for-black-box-models)
- 逻辑回归可解释性 → [白盒模型](#white-box-ml-models-linear--logistic)
- 损失函数 → [损失函数总览](#loss-functions-quick-view)
- MSE（均方误差）→ [讲义 6 > 回归](#regression-basics-mse)
- 梯度下降（学习率）→ [讲义 6 > 优化](#gradient-descent)
- 逻辑回归（Sigmoid）→ [讲义 6 > 分类](#logistic-regression)
- 感知器/MLP → [神经网络基础](#neural-net-basics-perceptron-and-mlp)
- 优化（SGD/Momentum/RMSProp/Adam）→ [优化算法](#optimization-algorithms-and-schedules)
- 过拟合/欠拟合 → [正则化/泛化](#regularization-and-generalization)
- PDP/ICE → [可解释性 XAI](#explainability-xai-for-black-box-models)
- ROC/PR 曲线 → [指标](#classification-metrics-and-curves)
- RNN/LSTM/GRU → [序列模型](#rnnlstmgru-basics)
- SHAP 基本概念 → [可解释性 XAI](#explainability-xai-for-black-box-models)
- Softmax + 交叉熵梯度 → [公式](#formula-quick-reference)
- 感知器/XOR/激活（ReLU）→ [讲义 7 > 基础](#xor-and-activations)
- 卷积/滤波器/核 → [讲义 7 > CNN](#cnn-convolution-basics)

---

## 讲义概要（基于幻灯片内容）

### 讲义 6：AI 白盒（来自讲义）

#### AI 基础与问题刻画
- 自适应系统：基于观测做出理性行动；智能体以最大化期望效用为目标。
- 预测任务要素：特征（输入）、模型 f(.)、预测值 ŷ、真实标签（Target）。示例：用振动、年龄、压力、温度预测机器是否故障。

一句话理解：“看 → 想 → 做”。我们先观测到数字（特征），用一个“配方/步骤”（模型）算出答案（预测），再和真实答案对比（标签）。
记住三点：
- 输入是你量到的东西；输出是你想知道的结果。
- 模型就是一串公式/小步骤的组合。
- 好模型=在有答案的数据上，误差尽量小。

<a id="regression-basics-mse"></a>
#### 回归基础（MSE）
- 线性模型：y = m x + b；多特征形式 y = w·x + b。
- 损失：均方误差 MSE，平均平方残差，避免正负抵消并加大大误差惩罚。

大白话：
- 画一条最贴近点的直线。好的直线=每个点到线的“竖直距离”的平方尽量小。

迷你例子：
- 真实 y: [2,3]；预测 y: [1,4] → 误差: [+1, −1]；平方: [1,1]；MSE = (1+1)/2 = 1。

常见坑：
- 把输入 x 和输出 y 搞反，应是 ŷ = w·x + b。
- 忘了“平方”，导致正负误差互相抵消。

新手模式：回归与 MSE — 超简单版
- 目标：画一条能最好地用 x 预测 y 的直线。
- 单点误差 = 真实 − 预测；平方让“大错”更疼。
- MSE = 所有平方误差的平均，越小越好。

1 分钟练习：
- 数据 (x,y)=(1,2),(2,3)，若直线 ŷ = x + 0。
- 预测 [1,2] → 误差 [1,1] → 平方 [1,1] → MSE=1。
- 上移 1：ŷ = x + 1 → 预测 [2,3] → 误差 [0,0] → MSE=0（此玩具例完美）。

注意：
- 量纲重要（米/美元等），必要时做特征缩放。
- 异常值会“主宰”MSE，必要时考虑 MAE/Huber。

<a id="gradient-descent"></a>
#### 梯度下降
- 迭代优化：按负梯度方向更新以降低损失；关键超参为学习率 α（过大易越过最小值，过小收敛慢）。
- 多特征情形对应更高维参数空间，同理可行。

类比：大雾中下山，你摸到最陡的下坡方向，小步走下去，反复直到地势变平。
小建议：
- 损失忽上忽下=步子太大（减小学习率）。
- 损失很慢才降=步子太小（稍微增大学习率）。

<a id="logistic-regression"></a>
#### 白盒分类（逻辑回归、决策树）
- Sigmoid 将分数映射到 [0,1]；逻辑回归=线性打分+Sigmoid，输出概率。
- 决策树通过贪心选择使子节点“更纯”的划分：
  - 不纯度度量：熵、Gini；信息增益 = 父不纯度 − 子不纯度加权和。
  - 训练：枚举特征与切分点，评估纯度并选择最佳划分，递归构建；优点可解释，缺点易过拟合/高方差。

  超直白：
  - 逻辑回归：把特征加权求和，再用 Sigmoid 把结果压到 0~1 之间，当作“概率”。
  - 决策树：不断问“是/否”的问题，把数据分成越来越“单一”的小堆。

  极简 IG 算法题：
  - 父不纯度=0.8；切分后两个子集不纯度=0.5（权重0.6）和0.2（权重0.4）。
  - 子集加权=0.6×0.5+0.4×0.2=0.38；信息增益 IG=0.8−0.38=0.42（不错）。

锚点： [回归](#regression-basics-mse) • [梯度下降](#gradient-descent) • [逻辑回归](#logistic-regression) • [决策树](#decision-trees-id3c45-basics)

### 讲义 7：神经网络（来自讲义）

#### 动机与特征学习
- 结构化 vs 非结构化数据：前者特征显式，后者（图像/音频/文本）特征隐含且复杂。
- 从手工特征工程转向特征学习：模型直接从原始数据学习有效表示。

<a id="xor-and-activations"></a>
#### 感知器、XOR 与激活函数
- 感知器是线性分类器：z = w·x + b，仅能解决线性可分问题。
- XOR 需要非线性边界 → 引入非线性激活（如 ReLU）与多层结构。

直觉版：
- 一条直线分不开“异或”这种交错点阵。多加几层+用“弯曲”的激活函数，边界就能弯起来。

<a id="neural-net-basics-perceptron-and-mlp"></a>
#### MLP 结构与训练
- 结构：输入层 → 隐藏层（学中间表示）→ 输出层。
- 通用逼近：至少一层隐藏层 + 非线性，MLP 可逼近任意连续函数。
- 训练流程：前向传播 → 计算损失 → 反向传播（求梯度）→ 优化器更新权重。

不要微积分也能懂的反传：
- 问“是谁导致了误差”和“导致了多少”。把“锅”一层层往前传，每个权重朝改进方向挪一点点。

排错小流程：
- 先在极小 batch（如 8 条）上“过拟合”。过不去=90% 是 shape、学习率、激活/损失不匹配的问题。

<a id="cnn-basics"></a>
#### CNN 卷积基础
- 动机：MLP 处理图像会破坏空间结构且参数量暴涨；CNN 利用局部性与权值共享。
- 卷积：小核/滤波器在图像上滑动生成特征图；滤波器学到边缘、纹理等局部模式。

3×3 小例子：
- 3×3 图像块 × 3×3 滤波器 → 9 个数相乘再相加，得到 1 个输出；把滤波器平移一格重复。
- 一个滤波器就得到一张“特征图”。

为什么好用：
- 同一套参数到处找同样的模式，参数更少、更会举一反三。

ASCII 滑窗相乘相加（2×2 卷积核扫 4×4 输入，步幅 1）：

```
输入 (4x4)                 卷积核 (2x2)
+--+--+--+--+              +--+--+
| 1| 2| 3| 4|              | a| b|
+--+--+--+--+              +--+--+
| 5| 6| 7| 8|      配合    | c| d|
+--+--+--+--+              +--+--+
| 9|10|11|12|
+--+--+--+--+
|13|14|15|16|
+--+--+--+--+

第一个输出(左上)：1*a + 2*b + 5*c + 6*d
向右平移 1 格 → 用 (2,3,6,7) 计算下一个输出。
```

锚点： [感知器/XOR](#xor-and-activations) • [MLP](#neural-net-basics-perceptron-and-mlp) • [反向传播](#backpropagation-essentials) • [CNN](#cnn-basics)

---

## 讲义 6：AI 白盒

### “白盒” vs “黑盒”
- 白盒：结构可直接解释（规则、树、线性系数），易审计与解释。
- 黑盒：预测强但不透明（深度网络、集成提升），常需事后解释。
- 取舍：可解释性 vs 准确率 vs 复杂度；受应用/监管约束。

<a id="rule-based-and-expert-systems"></a>
### 规则系统与专家系统
- 组成：知识库（事实+规则）、推理引擎、工作记忆、解释模块。
- 规则形式：若（前件条件）则（后件结论），常为 Horn 子句。
- 推理方式：
  - 正向链（数据驱动）：从事实出发不断应用规则直至得到目标。
  - 反向链（目标驱动）：从查询/目标倒推，寻找能推出目标的规则并递归验证其前提。
- 优缺点：推理透明易解释；但知识获取困难、脆弱、对噪声/不确定性差。

<a id="decision-trees-id3c45-basics"></a>
### 决策树（ID3/C4.5 基础）
- 自顶向下贪心划分，使子集更“纯”。
- 划分准则：
  - ID3：信息增益（熵减少）。
  - C4.5：增益率（校正固有信息量），可处理连续特征/缺失值。
  - CART：分类用 Gini，不纯回归用 MSE。
- 停止/剪枝：最小叶样本、最大深度、后剪枝以防过拟合。
- 可解释性：根→叶路径即人可读规则。

三点速记：
1) 选择让“纯度”上升最多的切分。
2) 树太深容易背答案 → 限深/剪枝。
3) 每条路径就是一句 IF…THEN… 的人话规则。

ASCII 小图（极简示例）：

```
            +-----------+
            | 天气?     |
            +-----------+
              /       \
           晴天       雨天
            /           \
     +-----------+    +----------+
     | 湿度高?   |    | 刮风?    |
     +-----------+    +----------+
       /      \         /     \
     是      否        否      是
     否      是        是      否
```

<a id="entropy-and-information-gain"></a>
#### 熵与信息增益
- 熵：H(Y) = − Σ p(y) log2 p(y)
- 条件熵：H(Y|X) = Σ p(x) H(Y|X=x)
- 信息增益：IG(Y,X) = H(Y) − H(Y|X)
- 增益率（C4.5）：IG / IntrinsicInfo(X)，其中 IntrinsicInfo(X) = − Σ p(x) log2 p(x)

<a id="white-box-ml-models-linear--logistic"></a>
### 白盒 ML 模型（线性/逻辑）
- 线性回归：y ≈ w·x + b；系数可解释方向与强度。
- 逻辑回归：p(y=1|x) = σ(w·x + b)；每个特征的优势比（odds ratio）可解释。
- 优缺点：全局可解释、简单；但表达能力有限、对多重共线性敏感。

<a id="explainability-xai-for-black-box-models"></a>
### 黑盒模型的可解释性（XAI）
- 全局 vs 局部：
  - 全局：整体特征效应（置换重要性、PDP/ICE、全局 SHAP）。
  - 局部：实例级归因（LIME、SHAP、反事实）。
- 方法：
  - 置换重要性：打乱某特征观察性能下降。
  - PDP：改变单一特征，平均模型预测。
  - ICE：每个样本的 PDP 曲线，显露异质性。
  - LIME：在实例邻域采样，拟合加权线性代理模型。
  - SHAP：基于 Shapley 值的加性归因，具一致性。
  - 反事实：最小特征改动即可翻转预测，需可行、可操作、稀疏。
- 注意：强相关特征会误导；分布漂移会使解释失真；关注忠实度与稳定性。

怎么选：
- 看全局大势 → 置换重要性、PDP。
- 看某条样本“为啥这么判” → LIME（快）、SHAP（更忠实但慢）。
- 要“怎么改能翻盘” → 反事实（最小改动，翻预测）。

<a id="classification-metrics-and-curves"></a>
### 分类指标与曲线
- 混淆矩阵：TP、FP、TN、FN。
- 准确率 Acc = (TP+TN)/All —— 类别极不平衡时误导性强。
- 精确率 P = TP/(TP+FP)；召回率 R = TP/(TP+FN)；F1 = 2PR/(P+R)。
- ROC-AUC：排序度量；PR-AUC 在正负极不平衡时更有信息量。

一分钟地图：
- 类别很不平衡时，Accuracy 经常骗人。
- Precision：我判为正的里，有多少是真的？
- Recall：所有真的正里，我抓住了多少？
- F1：当你想要一个折中分数时用它。

<a id="responsible-ai-fairness-transparency"></a>
### 负责任的 AI（公平与透明）
- 公平性定义：群体统计一致（DP）、机会均等（EOpp）、机会均衡（EOdds）。
- 偏差来源：数据、标签、代理特征；缓解：重加权、约束、后处理。
- 文档化：模型卡片、数据说明；治理：审计可追踪。

---

## 讲义 7：神经网络

<a id="neural-net-basics-perceptron-and-mlp"></a>
### 神经网络基础：感知器与 MLP
- 感知器：线性阈值单元；遇到非线性可分问题（如 XOR）需隐藏层。
- MLP：线性变换层 + 非线性激活层的堆叠。
- 一隐层前向：
  - h = φ(W1 x + b1)
  - ŷ = f(W2 h + b2)（多分类常用 softmax，二分类常用 sigmoid）

<a id="activation-functions"></a>
### 激活函数
- Sigmoid：σ(z) = 1/(1+e^(−z))；导数 σ(1−σ)；饱和易梯度消失。
- Tanh：零中心；导数 1−tanh^2。
- ReLU：max(0,z)；不饱和但可能“神经元死亡”。
- LeakyReLU/ELU/GELU：缓解死亡、改善梯度。

怎么选：
- 先用 ReLU。死神经元多或梯度乱 → 尝试 LeakyReLU/GELU。
- RNN 的门里常见 sigmoid/tanh（不是拿来堆很深的）。

<a id="loss-functions-quick-view"></a>
### 损失函数（速览）
- 回归：MSE、MAE、Huber。
- 二分类：BCE（对数损失）。
- 多分类：Softmax + 交叉熵。

<a id="backpropagation-essentials"></a>
### 反向传播要点
- 通过计算图应用链式法则求 ∂L/∂θ。
- 线性层 y = Wx + b，若上游梯度 g = ∂L/∂y：
  - ∂L/∂W = g x^T；∂L/∂x = W^T g；∂L/∂b = g（对批求和）。
- Softmax + CE 输出梯度简化为 p − y（y 为 one-hot）。

理解 p − y：
- 真实类给了 0.9（y=1）→ 梯度=0.9−1=−0.1：只需小修小补。
- 真实类给了 0.1 → 梯度=0.1−1=−0.9：需要大力把真实类分数推高。

<a id="optimization-algorithms-and-schedules"></a>
### 优化算法与学习率日程
- SGD：θ ← θ − η ∇L
- Momentum：v ← βv + (1−β)∇L；θ ← θ − η v
- RMSProp：按参数自适应步长，跟踪平方梯度 EMA。
- Adam：一阶/二阶矩 EMA + 偏差校正；常用默认。
- 学习率策略：阶梯/平台衰减、余弦、预热；学习率需精心调优。

开箱即用组合：
- Adam + LR≈1e−3（分类）通常能跑起来；视觉任务 1e−4~1e−3 常见。
- 卡住不降 → 降 LR（平台衰减）或用余弦退火 + 预热。

<a id="regularization-and-generalization"></a>
### 正则化与泛化
- L2（权重衰减）：加 λ||w||^2，收缩权重。
- L1：稀疏化，具特征选择作用。
- Dropout：随机置零激活；推理时缩放。
- 早停：监控验证集指标提前停止防过拟合。
- BatchNorm：稳定/加速训练，兼具轻微正则效果。
- 数据增强：翻转/裁剪/噪声；视觉中有 mixup/cutmix。

过拟合小体征：
- 训练降、验证升 → 过拟合。
- 解法：更多数据、更强增强、更强正则（L2/Dropout）、更早停、更小模型。

### CNN 基础
- 卷积：局部感受野与权值共享。
- 一维输出尺寸：out = floor((n + 2p − k)/s) + 1；参数量 = k*k*C_in*C_out + C_out（含偏置）。
- 池化：下采样（最大/平均）以获得平移等不变性。

<a id="rnnlstmgru-basics"></a>
### RNN/LSTM/GRU 基础
- RNN：h_t = φ(W_x x_t + W_h h_{t−1} + b)。
- 问题：长序列上梯度消失/爆炸。
- LSTM/GRU：门控机制以保留/遗忘信息。
- 缓解：门控、残差、层归一、梯度裁剪、缩短 BPTT 窗口。

<a id="vanishingexploding-gradients"></a>
### 梯度消失/爆炸
- 成因：雅可比连乘幅值持续 <1 或 >1；激活饱和；深层未归一化堆叠。
- 对策：ReLU/GELU、He/Xavier 初始化、归一化、残差、门控、裁剪。

<a id="practical-training-tips"></a>
### 实操训练技巧
- 标准化输入；输出/激活/损失要匹配。
- 分类可从 Adam, LR≈1e−3 起步并调优；可用 LR Finder 或小网格搜索。
- 核对张量形状，监控梯度，先在很小的 batch 上“过拟合”做健检。
- 固定随机种子；记录版本；合理划分训练/验证/测试。

---

## 可能考题与示范答案

1）解释正向链推与反向链推，并给出小示例。
- 正向：事实 F={A,B}，规则 R1: A∧B→C，R2: C→D → 推得 C，再推得 D。
- 反向：为证明 D 先需 C（R2），为证明 C 需 A、B（R1）；检查事实是否满足。
- 适用：正向用于数据流/诊断；反向用于问答/专家咨询。

2）计算一次划分的信息增益。
- 若 Y 的 p+=0.6, p−=0.4：H(Y)=−0.6log2 0.6 −0.4log2 0.4≈0.971。
- 特征 X 将数据分为权重 2/3 与 1/3 的两组，其熵分别 0.811 与 0.0 → H(Y|X)=0.541。
- IG(Y,X)=0.971−0.541=0.430。

小提示：背下“父不纯度 − 子不纯度加权和”。IG 大=问题问得好。

3）为何决策树被视为白盒？与神经网络的利弊对比。
- 根到叶的路径即规则，易解释。优点：透明、可处理混合类型、预处理少。缺点：高方差、轴对齐划分、在复杂模式上准确率不如集成/NN。

4）对比 LIME 与 SHAP，何时选择哪一个？
- LIME：局部线性代理，速度快但一致性弱；适合快速实例解释。
- SHAP：Shapley 一致的加性归因，更忠实但更耗时；适合审计/重要度分析。

5）推导一隐层 MLP 的 Softmax+CE 梯度。
- 输出层梯度 δ_out = p − y。
- W2 梯度：δ_out h^T；b2 梯度：对 batch 的 δ_out 求和。
- 隐层梯度：δ_h = (W2^T δ_out) ⊙ φ′(z1)；进而 W1 梯度：δ_h x^T；b1 梯度：δ_h。

6）解释梯度消失并给出两种缓解方式。
- 连续乘积使梯度收缩；可用 ReLU/GELU、残差连接、良好初始化、归一化，或序列任务用 GRU/LSTM。

7）给定混淆矩阵，计算 P、R、F1。
- P=TP/(TP+FP)；R=TP/(TP+FN)；F1=2PR/(P+R)。

快速例子：
- TP=8, FP=2, FN=4 → P=8/10=0.8，R=8/12≈0.667，F1≈0.727。

8）对比 L2 正则与 Dropout。
- L2：均匀收缩权重；保留全部特征；平滑损失地形。
- Dropout：随机丢弃激活；相当于模型集成；正则更强。

9）卷积层的输出尺寸与参数量计算。
- 例：输入 32×32×3，k=3，p=1，s=1，C_out=64 → 输出 32×32×64；参数 = 3×3×3×64 + 64 = 1,792。

10）为何 BatchNorm 有助于训练。
- 稳定激活分布，允许更大学习率，平滑优化；兼具轻度正则作用。

11）何为反事实解释，其约束为何？
- 以最小特征改变翻转预测；应可实现、可操作且稀疏。

12）树剪枝中的前向与后向思路（高层）。
- 预剪枝（前向）：训练早期即设停条件。后剪枝（后向）：先长满再依据验证或代价复杂度剪枝。

---

<a id="formula-quick-reference"></a>
## 公式速查
- 熵：H(Y) = − Σ p(y) log2 p(y)
- 条件熵：H(Y|X) = Σ_x p(x) H(Y|X=x)
- 信息增益：IG = H(Y) − H(Y|X)
- 增益率：IG / (−Σ p(x) log2 p(x))
- Sigmoid：σ(z)=1/(1+e^(−z))；σ′=σ(1−σ)
- Tanh：tanh′=1−tanh^2
- ReLU：max(0,z)
- Softmax：p_i = exp(z_i)/Σ_j exp(z_j)
- CE 梯度（softmax）：∂L/∂z = p − y
- 线性层：∂L/∂W = g x^T；∂L/∂x = W^T g；∂L/∂b = g
- L2 正则：加 λ||w||^2 → ∂L/∂w += 2λw（或按权衰约定为 λw）
- 卷积输出（1D）：out=floor((n+2p−k)/s)+1；参数=k*k*C_in*C_out + C_out

---

## 术语表（快速查）
- BCE：二元交叉熵。CE：交叉熵。AUC：ROC/PR 曲线下面积。
- PDP/ICE：偏依赖图 / 个体条件期望曲线。
- LIME/SHAP：局部可解释代理 / Shapley 加性解释。
- BPTT：通过时间的反向传播。
- BN：BatchNorm。WD：Weight Decay（权重衰减）。

---

## 考试使用建议
- 先用“快速索引”和“术语表”跳转；英文/中文关键词都可 Ctrl+F 搜索。
- 计算题直接跳“公式速查”。
- 比较/论述题先看“预测考题与示范答案”。
