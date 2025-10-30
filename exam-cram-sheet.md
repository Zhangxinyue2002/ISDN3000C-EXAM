# ISDN3000C Exam Cram Sheet (EN + 中文)

Use with the full guide: see `exam-cheatsheet.bilingual.md`.

## Core formulas 核心公式

- Linear regression 线性回归
  - ŷ = w·x + b
  - MSE = (1/n) Σ_i (y_i − ŷ_i)^2
  - ∂MSE/∂w = −(2/n) X^T (y − ŷ), ∂MSE/∂b = −(2/n) Σ_i (y_i − ŷ_i)
- Logistic regression 逻辑回归
  - p = σ(z) = 1/(1+e^(−z)), z = w·x + b
  - BCE = −(1/n) Σ_i [ y_i ln p_i + (1−y_i) ln(1−p_i) ]
  - ∂BCE/∂w = (1/n) X^T (p − y), ∂BCE/∂b = (1/n) Σ_i (p_i − y_i)
- Decision tree 决策树
  - 熵 Entropy: H(S) = − Σ_c p(c) log₂ p(c)
  - 信息增益 IG(S, A) = H(S) − Σ_v (|S_v|/|S|) H(S_v)
- Metrics 指标
  - Acc = (TP+TN)/(TP+TN+FP+FN)
  - Prec = TP/(TP+FP), Rec = TP/(TP+FN), F1 = 2PR/(P+R)
- Backprop 反向传播（线性层）
  - δ_L = ∂L/∂z_L；dW = δ x^T；db = δ
  - δ_l = (W_{l+1}^T δ_{l+1}) ⊙ f'(z_l)
- CNN quick CNN 速记
  - Conv out: H_out = ⌊(H + 2P − K)/S⌋ + 1, W_out 同理
  - Params: K×K×C_in×C_out + C_out（若带 bias）
  - Pooling 无参数；常用 2×2, stride=2

## Tiny examples 迷你例子

- IG quick split: dataset class counts S: {+3, −1} → H(S)=0.811. Attribute A splits into S1 {+2,−0} H=0, S2 {+1,−1} H=1 → IG = 0.811 − (2/4·0 + 2/4·1) = 0.311.
- 3×3 input, 3×3 kernel, stride=1, padding=0 → output 1×1。2 个 3×3 kernel → 输出通道数 2。

## Predicted questions 考点预测（速答版）

- MSE vs MAE: MSE 放大异常值、可导且平滑；MAE 抗异常值、梯度常数。选择看是否怕 outlier。
- Entropy vs Gini: 排序常一致；Gini 计算略快，Entropy 信息解释更直观。考试写 IG 步骤即可。
- Vanishing gradient: 深层或 σ/tanh 饱和；用 ReLU/LeakyReLU、良好初始化、归一化、较浅网络或残差。
- Why padding/stride: 控制空间尺寸、保边界信息、调节计算量与感受野。
- Threshold choice: 根据代价权衡 Precision/Recall；医疗偏召回，垃圾邮件偏精度。
- CNN params: 记 K×K×C_in×C_out + C_out。

## ASCII minis 速记图

- Decision tree 决策树

  Root A?
  ├─ v1 → [pure]
  └─ v2 → split B?
         ├─ b1 → [+]
         └─ b2 → [−]

- Confusion matrix 混淆矩阵

            Pred +   Pred −
  Actual +    TP        FN
  Actual −    FP        TN

- ROC / PR intuition 直觉

  ROC: TPR vs FPR（阈值从高到低移动，曲线向左上越好，AUC 越大越好）
  PR:  Precision vs Recall（正类稀少更看 PR）

## Super‑short reminders 超短提示

- 先缩放特征，再用 GD；学习率过大会震荡，过小很慢。
- 树要剪枝避免过拟合；用验证集挑深度/最小样本数。
- 指标看类不平衡：Acc 可能误导，报告 P/R/F1/ROC‑AUC。
- CNN 中 padding=“same” 常用于保尺寸；多用小核堆叠增感受野。
