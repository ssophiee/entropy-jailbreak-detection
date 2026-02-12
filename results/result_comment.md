## Prompt-Reading Entropy Aggregation and Detection Metrics

To detect harmful prompts, we compute the token-level predictive entropy while the model reads the prompt under teacher forcing. This produces an entropy trace over token positions, which we aggregate into scalar features. We evaluate multiple aggregation strategies:

- Mean / Median: global central tendency of uncertainty over the prompt.
- Trimmed Mean: robust central tendency after removing extreme values, mitigating the effect of entropy spikes.
- Min / Max / Range: extremal behavior, capturing isolated high- or low-uncertainty regions.
- Standard Deviation (std): variability of uncertainty across positions.
- Quantiles (p10, p90): lower and upper distributional tails, capturing persistent low- or high-uncertainty regimes.
- First/Last Mean: average entropy over early vs. late prompt segments, probing position-dependent effects.
- AUC (sum of entropies): total accumulated uncertainty across the prompt.
- Fraction Above Quantile: proportion of positions exceeding a high-entropy threshold, measuring “spikiness.”
- Slope: linear regression coefficient of entropy over token index, capturing systematic upward or downward trends in uncertainty across the prompt.

For evaluation, each aggregated feature is treated as a continuous detection score (harmful = positive class). We report **Accuracy** (median threshold, untuned), **AUROC** (ranking-based separability), **Average Precision** (AP) (precision-recall tradeoff), **TPR at 1% and 0.1% FPR** (extreme low-false-positive regime relevant to safety), as well as ECE and Brier score after monotonic min–max normalization to assess probabilistic calibration.

Among all aggregations, the entropy `slope` achieves the strongest performance (AUROC ≈ 0.80), substantially outperforming central-tendency measures such as `mean` or `median`. This indicates that harmful prompts are not primarily characterized by uniformly high or low uncertainty, but rather by a systematic drift in uncertainty as the prompt unfolds. In other words, adversarial or jailbreak prompts tend to induce a progressive change in model confidence across token positions, a structural signal that is not captured by static summary statistics. The slope therefore captures a dynamic property of prompt construction, explaining its superior discriminative power.


|key       |accuracy   |auroc        |auroc_flipped|average_precision|tpr@1%fpr  |tpr@0.1%fpr   |ece        |brier       |
|----------|-----------|-------------|-------------|-----------------|-----------|--------------|-----------|------------|
|mean      |0.56       |0.5424       |0.4576       |0.531954         |0          |0             |0.220325   |0.307888    |
|std       |0.32       |0.2516       |0.7484       |0.36415          |0          |0             |0.405809   |0.405359    |
|min       |0.54       |0.625          |0.375          |0.580193         |0          |0             |0.441249   |0.428013    |
|max       |0.25       |0.1732       |0.8268       |0.348634         |0          |0             |0.515        |0.486412    |
|median    |0.6        |0.6048       |0.3952       |0.598238         |0          |0             |0.102913   |0.259855    |
|p10       |0.66       |0.6492       |0.3508       |0.602419         |0          |0             |0.352401   |0.350826    |
|p90       |0.46       |0.4252       |0.5748       |0.428051         |0          |0             |0.305508   |0.325887    |
|trimmed_mean|0.58       |0.6044       |0.3956       |0.577985         |0          |0             |0.136621   |0.266742    |
|first_mean|0.36       |0.3058       |0.6942       |0.389624         |0          |0             |0.2869     |0.352269    |
|last_mean |0.68       |0.755          |0.245          |0.723681         |0.04       |0.04          |0.167835   |0.236905    |
|auc       |0.32       |0.2296       |0.7704       |0.359268         |0          |0             |0.371315   |0.394606    |
|slope     |0.7        |0.7972       |0.2028       |0.825832         |0.34       |0.34          |0.143297   |0.21778     |
|frac_above_q|0.56       |0.5674       |0.4326       |0.569861         |0          |0             |0.175248   |0.318446    |
|range     |0.2        |0.1508       |0.8492       |0.33955          |0          |0             |0.508977   |0.494424    |


---
## Update – Entropy Dynamics

To better capture structural changes in uncertainty across the prompt, we introduce additional trend- and dynamics-based aggregations:

- **Delta End** (delta_end): difference between final and initial entropy values, measuring net uncertainty drift.

- **Delta Segment** (delta_seg): difference between the mean entropy of the last and first prompt segments, providing a more robust early–late contrast.

- **Spearman Correlation** (spearman_rho): rank-based correlation between entropy and token index, capturing monotonic (not necessarily linear) trends.

- **Total Variation** (total_variation): sum of absolute entropy differences, measuring volatility or jaggedness of the entropy trace.

- **Monotonicity Ratio** (monotonicity_up): fraction of upward entropy changes relative to total variation, indicating directional consistency.

- **Peak Position** (peak_pos): normalized token index at which entropy is maximal, identifying where uncertainty concentrates within the prompt.

Across these dynamics-aware metrics, early–late drift measures emerge as the strongest signals. In particular, `delta_seg` (AUROC ≈ 0.83) and `slope` (AUROC ≈ 0.80) outperform all level-based statistics (e.g., mean, median, quantiles), confirming that harmful prompts are characterized not by uniformly higher entropy, but by a systematic increase in uncertainty as the prompt unfolds. The strong performance of `spearman_rho` further indicates that the signal is monotonic rather than strictly linear, reinforcing the interpretation that adversarial prompts exhibit progressive structural escalation.

|key       |accuracy   |auroc        |auroc_flipped|average_precision|tpr@1%fpr  |tpr@0.1%fpr   |ece        |brier       |
|----------|-----------|-------------|-------------|-----------------|-----------|--------------|-----------|------------|
|delta_end |0.62       |0.6744       |0.3256       |0.719416         |0.06       |0.06          |0.116942   |0.227506    |
|delta_seg |0.72       |0.8312       |0.1688       |0.818387         |0.2        |0.2           |0.158154   |0.190976    |
|spearman_rho|0.72       |0.7896       |0.2104       |0.808964         |0.18       |0.18          |0.0983949  |0.197361    |
|total_variation|0.26       |0.1592       |0.8408       |0.338612         |0          |0             |0.428562   |0.434513    |
|monotonicity_up|0.64       |0.664          |0.336          |0.730723         |0.1        |0.1           |0.0965716  |0.233678    |
|peak_pos  |0.62       |0.6626       |0.3374       |0.714205         |0.1        |0.1           |0.148935   |0.251208    |
