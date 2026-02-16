== Post-Analysis for Dimension Reduction

To improve cluster balance and interpretability, feature selection analysis was conducted to identify the most discriminative variables.

=== Feature Categorization

Features were systematically separated into boolean and continuous types:

Feature partitioning:

$F_"bool" = {f_j : f_j in {0, 1} and f_j != "labels"}$

$F_"cont" = F without (F_"bool" union {"labels"})$

=== Temporal Feature Analysis

Date-encoded features were examined for variability:

Temporal analysis: $Delta t_i = t_"admission"^i - t_"symptoms"^i$

Result: $Delta t_i = 0 forall i$, thus date features removed.

All differences equaled zero, indicating patients were admitted on the day of first symptom onset. Consequently, both date features were removed from the continuous feature set as they provided no discriminative power.

=== Continuous Variable Discrimination Analysis

Three metrics assessed the discriminative power of continuous features:

*Between-Cluster Mean Separation:* Measures how far apart cluster centers are (larger is better):

Between-cluster mean separation:

$sigma_"between"(f_j) = sqrt(1/K sum_(k=1)^K (mu_(k j) - overline(mu)_j)^2)$

where $mu_(k j) = "mean"({f_(i j) : L_i = k})$ and $overline(mu)_j = 1/K sum_(k=1)^K mu_(k j)$

*Within-Cluster Standard Deviation:* Measures cluster tightness (smaller is better):

Within-cluster standard deviation:

$overline(sigma)_"within"(f_j) = 1/K sum_(k=1)^K sigma_(k j)$

where $sigma_(k j) = sqrt(1/(n_k - 1) sum_(i: L_i = k) (f_(i j) - mu_(k j))^2)$

*Discriminative Ratio:* Combines both metrics to assess overall separation quality:

Discriminative ratio:

$rho(f_j) = sigma_"between"(f_j) / overline(sigma)_"within"(f_j)$

Results revealed:

```
Feature                  Discriminative Ratio   Interpretation
oxygen_saturation        0.016                  Almost total overlap
fever_temperature        0.045                  Almost total overlap
age                      0.280                  Partial separation
nationality_numeric      8.930                  Suspiciously high
```

The interpretation scale used:
- < 0.05: Centers tiny compared to spread
- 0.05–0.10: Almost total overlap
- 0.10–0.30: Partial separation
- 0.30–0.50: Clear but overlapping
- 0.50–1.00: Strong separation
- > 1.00: Very strong/suspicious

*Analysis Conclusions:*

*Oxygen saturation* and *fever temperature* showed poor discrimination (ratios < 0.05), with cluster centers barely separable relative to within-cluster variation. However, both features hold critical medical significance.

*Age* demonstrated moderate discriminative power (ratio: 0.28), with clusters showing partial separation by patient age.

*Nationality_numeric* exhibited suspiciously high separation (ratio: 8.93). This occurred because hot-encoded nationality labels lack true numerical ordering—the numeric codes are arbitrary identifiers rather than meaningful continuous values.

Given the medical importance of vital signs despite their low statistical discrimination, we decided to combine oxygen_saturation and fever_temperature into a single feature using PCA rather than discarding them entirely.

=== Boolean Feature Discrimination Analysis

Two complementary metrics evaluated boolean (symptom and comorbidity) features:

*Delta P (Effect Size):* Measures the maximum difference in symptom prevalence across clusters:

Effect size (Delta P):

$Delta p(f_j) = max_k p_(k j) - min_k p_(k j)$

where $p_(k j) = 1/(n_k) sum_(i: L_i = k) f_(i j)$ is the prevalence in cluster $k$.

*Cramér's V (Association Strength):* Quantifies statistical association between feature and cluster assignment:

Cramér's V statistic:

$V = sqrt(chi^2 / (n dot (k - 1)))$

where:
- $chi^2 = sum_(i, j) ((O_(i j) - E_(i j))^2) / E_(i j)$ is the chi-squared statistic
- $n$ = total sample size
- $k = min(r, c)$ for contingency table with $r$ rows and $c$ columns

The Cramér's V interpretation scale:
- < 0.05: No discrimination
- 0.05–0.10: Weak
- 0.10–0.20: Moderate
- 0.20–0.30: Strong
- > 0.30: Very strong

Statistical analysis identified the top discriminative features as:
```
pcr_result, history_of_fever, fatigue_malaise, sore_throat
```

*Refined Feature Selection:*

Despite strong statistical associations, *pcr_result* and *history_of_fever* were excluded from the final feature set. PCR result represents the diagnostic outcome rather than a symptom predictor, and history of fever largely overlaps with the fever_temperature measurement.

*Headache* was added based on medical domain knowledge despite moderate statistical scores, as it represents a distinctive COVID-19 symptom pattern.

*Final selected boolean features:*
$F_"selected" = {"fatigue_malaise", "sore_throat", "headache"}$