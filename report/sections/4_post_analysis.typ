== Post-Analysis for Dimension Reduction

In order to improve cluster balance, interpretability and also in accordance with theory. Feature selection analysis was conducted to identify the most discriminative variables.

=== Feature Categorization

At first glance would be easy to measure difference between minimum and maximum features. However, due to their different scales of numbers and there meaning it is not possible to classify a difference without standardizing the output. 

For example:

$"age_diff" = 98 = max(100) - min(2)$

$"fever_diff" = 5.3 = max(40.1) - min(34.8)$

As seen in the previous example, the difference is too to be able to take into account. This it's because the number range it doesn't match. Additionally, not all values can be subtracted. Most of the dataset it's just boolean values.

In order to avoid this, I systematically separated analysis into boolean and continuous types:

Where:

$ F_"bool" = {f_j : f_j in {0, 1} and f_j != "labels"} $

$ F_"cont" = F without (F_"bool" union {"labels"}) $

=== Temporal Feature Analysis

A special case for date encoded features was applied in order to examined variability. Based on previous observations, instead of dealing with temporal causation it was better to remove the temporality of values by subtracting the start and end dates.

However, all differences equaled zero, indicating patients were admitted on the day of first symptom onset. Consequently, both date features were removed from the continuous feature set as they provided no discriminative power.

=== Continuous Variable Discrimination Analysis

Following the previous idea that not a single value could account for scaling. Two separate methods were implement and then aggregated

*Between-Cluster Mean Separation:* Measures how far apart cluster centers are (larger is better). Where for each feature I:

- Compute the mean of that feature inside each cluster
- Take the standard deviation of those means

**Ej**: Each clusters mean differ from each other X...

*Within-Cluster Standard Deviation:* Measures cluster tightness (smaller is better). Where for each feature:

- Compute the std inside each cluster
- Average those std values across clusters

**Ej**: Inside each cluster, patients’ features differ by: X....

After each implementation was computed, a *discriminative Ratio:* was used combine both metrics to assess overall separation quality:

Discriminative ratio:

$ "ratio"(f_i) = "Between-Cluster"(f_i) / "Within-Cluster"(f_i) $

Were the following interpretation scale used:
- < 0.05: Centers tiny compared to spread
- 0.05–0.10: Almost total overlap
- 0.10–0.30: Partial separation
- 0.30–0.50: Clear but overlapping
- 0.50–1.00: Strong separation
- > 1.00: Very strong/suspicious

After evaluation, the most influential values we're sorted revealing that:

```
Feature                  Discriminative Ratio   Interpretation
oxygen_saturation        0.016                  Almost total overlap
fever_temperature        0.045                  Almost total overlap
age                      0.280                  Partial separation
nationality_numeric      8.930                  Suspiciously high
```

=== Analysis Conclusions:

*Oxygen saturation* and *fever temperature* showed poor discrimination (ratios < 0.05), with cluster centers barely separable relative to within-cluster variation. However, both features hold critical medical significance.

*Age* demonstrated moderate discriminative power (ratio: 0.28), with clusters showing partial separation by patient age.

*Nationality_numeric* exhibited suspiciously high separation (ratio: 8.93). This occurred because hot-encoded nationality labels lack true numerical ordering. Numeric codes are arbitrary identifiers rather than meaningful continuous values.

Given the medical importance of vital signs despite their low statistical discrimination, I decided to combine oxygen_saturation and fever_temperature into a single feature using PCA rather than discarding them entirely.

=== Boolean Feature Discrimination Analysis

The same analysis can't be applied to boolean value. In order to evaluate these metrics we need to evaluate the distribution against the wholesome.

*Delta P (Effect Size):* Measures the maximum difference in symptom prevalence across clusters or what is the probability of being 1 changing across clusters?

*Cramér's V (Association Strength):* Quantifies statistical association between feature and cluster assignment where:

$ V = sqrt(chi^2 / (n dot (k - 1))) $

Assuming:
- $chi^2 = sum_(i, j) ((O_(i j) - E_(i j))^2) / E_(i j)$ is the chi-squared statistic
- $n$ = total sample size
- $k = min(r, c)$ for contingency table with $r$ rows and $c$ columns

This results, can be evaluated using the following interpretation scale:

- < 0.05: No discrimination
- 0.05–0.10: Weak
- 0.10–0.20: Moderate
- 0.20–0.30: Strong
- > 0.30: Very strong

The #link(<cramer>)[following listing], shows a practical example:

#figure(
  caption: [Initial Cluster visualization],
  ```
  If I know the cluster label, how much can I guess whether this feature is 0 or 1?
  
  Ej: Assuming the the spread of values, we could say that each cluster has "x%" of the values.
  
  | Cluster | % Fever |
  | ------- | ------- |
  | 0       | 5%      |
  | 1       | 50%     |
  | 2       | 95%     |
  
  If cluster = 0 → guess no
  If cluster = 2 → guess yes
  ```)<cramer>

After applying this methodology, both values were taken into account and a simple filter function were both results had to greater than 0.05 it yield the following features:

```python
['pcr_result', 'history_of_fever', 'fatigue_malaise', 'sore_throat']

With the following values:

Feature            | Delta p | Cramér's
--------------------------------------
pcr_result         | 0.441   | 0.185
history_of_fever   | 0.285   | 0.081
fatigue_malaise    | 0.208   | 0.060
sore_throat        | 0.164   | 0.050
cough              | 0.137   | 0.043
muscle_aches       | 0.118   | 0.041
headache           | 0.094   | 0.035
hypertension       | 0.056   | 0.027
diarrhoea          | 0.055   | 0.023
smoking            | 0.047   | 0.029
```

=== Analysis Conclusions:

Despite strong statistical associations, *pcr_result* and *history_of_fever* were excluded from the final feature set. PCR result represents the diagnostic outcome rather than a symptom predictor while history of fever doesn't necessarily indicate a correlation with disease.

Additionally, *headache* was added based on common medical domain knowledge, despite it's statistical scores. This leaves us with the following set:

```python
["fatigue_malaise", "sore_throat", "headache"]
```

#pagebreak()

== Final conclusion

After analyzing individually both segments and based on the post-analysis findings, a reduced feature set was constructed combining statistically and clinically significant variables:

The final reduced feature set comprised five dimensions:
- *fatigue_malaise* (boolean symptom)
- *sore_throat* (boolean symptom)
- *headache* (boolean symptom)
- *age* (continuous demographic)
- *oxygen_fever* (continuous composite symptom)

#v(0.3cm)

If we plot in 2D we get the following diagram.

#figure(
  caption: [Reduced Feature Set 2D visualization],
  image("../assets/reduce_set.png", width:80%),
)

#v(0.3cm)

However, this representation is no better then the first representation that was computed. Even though we can see clearly where each representation starts and ends. There isn´t a clear breakdown between values.

#pagebreak()

If we try to plot the result in 3D. We can see that the lines are also separated by dept. This opens a new possibility to break not only by separation but rather by depth (PC1 axis).

#v(0.3cm)

#figure(
  caption: [Reduced Feature Set 3D visualization],
  image("../assets/3d_reduce_set.png", width: 80%),
)
