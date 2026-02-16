
= Second Clustering Attempt Using Reduced Dimensions

== Dimension Reduction Strategy

Based on the post-analysis findings, a reduced feature set was constructed combining statistically and clinically significant variables:

Dimension reduction: $bold(X)_"reduced" in RR^(n times 5)$ with features:

$F_"reduced" = F_"selected" union {"age"} union {"oxygen_fever"}$

where $"oxygen_fever"$ is the first principal component:

$f_"oxygen_fever" = bold(w)_1^T dot [(f_"oxygen", f_"temperature")^T - bold(mu)]$

with $bold(w)_1 = arg max_(||bold(w)|| = 1) "Var"(bold(X)_"vital" bold(w))$

Final scaling: $bold(X)' = "RobustScaler"(bold(X)_"reduced")$

The final reduced feature set comprised five dimensions:
- *fatigue_malaise* (boolean symptom)
- *sore_throat* (boolean symptom)
- *headache* (boolean symptom)
- *age* (continuous demographic)
- *oxygen_fever* (continuous vital sign composite)

== Visualization of Reduced Data

Three-dimensional PCA projection enabled visualization of the reduced feature space:

3D PCA projection: $bold(X)_"3D" = bold(X)_"reduced" bold(W)_3$ where $bold(W)_3 in RR^(5 times 3)$ contains the top 3 principal components.

Plot: ${bold(x)_i : L_i = k}$ for each cluster $k$, colored by cluster assignment.

== DBSCAN on Reduced Data

DBSCAN was applied to the reduced feature space with systematic hyperparameter optimization:

DBSCAN hyperparameter optimization:

$(epsilon^*, m^*) = arg max_((epsilon, m)) s(bold(X), "DBSCAN"(bold(X); epsilon, m))$

subject to: $K ≥ 2$ and $|N| < n$

Parameter grid:
- $epsilon in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}$ (neighborhood radius)
- $m in {5, 10, 15, 20}$ (minimum points)

where $N = {i : L_i = -1}$ is the noise set.

=== DBSCAN Results

The optimal DBSCAN configuration achieved:
- *Best parameters:* eps=0.5, min_samples=5
- *Silhouette score:* 0.301 (moderate separation)
- *Number of clusters:* 8
- *Noise points:* 0

DBSCAN successfully identified eight distinct patient groups without classifying any samples as noise. The moderate silhouette score (0.301) indicates overlapping but distinguishable clusters, suggesting genuine structure in the reduced feature space. Unlike BIRCH on full features, DBSCAN produced more balanced cluster sizes, enhancing practical interpretability.

== K-Means on Reduced Data

K-Means clustering was evaluated across multiple values of k to establish a centroid-based baseline:

```python
from sklearn.cluster import KMeans

best_score = -1
best_k = None
best_labels = None

k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=100)
    labels = kmeans.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)

    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels
```

=== K-Means Results

Silhouette scores across different k values:

```
k=2:  0.377
k=3:  0.368
k=4:  0.320
k=5:  0.342
k=6:  0.385
k=7:  0.412
k=8:  0.389
k=9:  0.418
k=10: 0.440 (best)
```

The optimal K-Means configuration identified:
- *Best k:* 10 clusters
- *Silhouette score:* 0.440 (moderate-good separation)

K-Means demonstrated progressive improvement with increasing k, achieving the highest silhouette score at k=10. This result outperformed DBSCAN (0.301) on the reduced feature set, suggesting that spherical cluster assumptions reasonably approximate the data structure after dimension reduction.

== BIRCH on Reduced Data

BIRCH was re-applied to the reduced feature space with hyperparameter optimization:

```python
param_distributions = {
    "threshold": uniform(0.1, 1.0),
    "branching_factor": randint(20, 100),
    "n_clusters": randint(2, 9)
}

results = compute_birch_with_hyperparams(X_reduced, param_distributions)
df_birch_redu_res = pd.DataFrame(results).sort_values(
    "silhouette", ascending=False
)

df_birch_reduced = df_birch_redu_res.iloc[0]
best_birch_labels = df_birch_reduced.labels
best_silhouette = df_birch_reduced.silhouette
```

=== BIRCH Reduced Results

BIRCH on reduced features achieved:
- *Silhouette score:* 0.977 (excellent, matching full-feature performance)
- *Number of clusters:* 3
- *Cluster distribution:*
  - Cluster 0: 503 patients (1.9%)
  - Cluster 1: 25,397 patients (96.8%)
  - Cluster 2: 337 patients (1.3%)

Notably, BIRCH maintained its exceptionally high silhouette score even after dimension reduction, but the cluster distribution remained severely imbalanced. The near-identical performance on both full and reduced feature sets suggests BIRCH is primarily identifying the same dominant patient subgroup (96.8% in Cluster 1) regardless of feature space dimensionality.

This persistent imbalance, despite excellent silhouette metrics, indicates that BIRCH may not be the optimal algorithm for this dataset when seeking balanced, clinically actionable patient stratification.

= Results and Discussion

== Algorithm Performance Comparison

The clustering experiments yielded contrasting results across algorithms and feature spaces:

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, left),
    [*Algorithm*], [*Features*], [*Silhouette*], [*Clusters*], [*Key Observation*],
    [BIRCH], [Full], [0.977], [3], [Highly imbalanced (96.8% in one cluster)],
    [DBSCAN], [Reduced], [0.301], [8], [Balanced clusters, no noise],
    [K-Means], [Reduced], [0.440], [10], [Best on reduced features],
    [BIRCH], [Reduced], [0.977], [3], [Same imbalance as full features],
  ),
  caption: [Clustering algorithm performance summary]
)

== Feature Importance Findings

The dimension reduction analysis revealed significant insights into COVID-19 symptom discrimination:

*Continuous Features:*
- *Age:* Emerged as the most discriminative continuous variable (ratio: 0.28), indicating partial cluster separation by patient age groups
- *Oxygen saturation & fever temperature:* Individually showed poor discrimination (ratios < 0.05), but their combination via PCA captured essential vital sign variation
- *Nationality:* Demonstrated statistically high separation (ratio: 8.93) but was excluded due to arbitrary hot-encoding rather than true numerical relationships

*Boolean Features:*
- *Top discriminative symptoms:* fatigue_malaise, sore_throat, and headache
- *Excluded despite statistical significance:* pcr_result (outcome rather than predictor) and history_of_fever (redundant with temperature measurement)
- Medical domain knowledge guided final feature selection, balancing statistical and clinical considerations

== Cluster Interpretation

Analysis of the initial BIRCH clustering with three clusters revealed distinct patient profiles:

*Cluster 0 (503 patients, 1.9%):* Younger patients (mean age: 38.5 years) with milder symptoms—lower fever (37.3°C), reduced oxygen saturation (93.9%), and lower rates of fever history (22.5%) and cough (15.9%). This group likely represents early-stage COVID or mild presentations.

*Cluster 1 (25,397 patients, 96.8%):* The dominant cluster with mean age 43.1 years, moderate symptoms including 51% fever history and 29.6% cough rate. This represents the standard COVID-19 patient profile.

*Cluster 2 (337 patients, 1.3%):* Youngest group (mean age: 34.9 years) with mixed symptom presentation—41.2% fever history, 24.3% cough, and 25.8% sore throat. May represent a distinct symptomatic subgroup.

== Methodological Insights

*Silhouette Score Limitations:* High silhouette scores (0.977 for BIRCH) do not guarantee clinically useful clustering. The severely imbalanced distribution suggests the metric captured one dominant group's homogeneity rather than meaningful patient stratification.

*Algorithm-Data Interaction:* BIRCH's hierarchical structure may be overly sensitive to the dataset's inherent imbalance (85% COVID-positive). DBSCAN and K-Means, operating on density and centroid principles respectively, produced more balanced groupings on reduced features.

*Feature Engineering Value:* Combining weakly discriminative but medically critical features (oxygen saturation and fever temperature) into a single PCA component preserved clinical information while reducing dimensionality.

*Scaling Considerations:* RobustScaler proved appropriate given outliers in vital sign measurements, though additional outlier investigation could further refine the analysis.

== Clinical Implications

The clustering results offer several potential applications for COVID-19 patient management:

*Risk Stratification:* The identification of distinct symptom profiles (particularly the mild symptom cluster) could support early triage decisions and resource allocation.

*Symptom Monitoring:* The key discriminative features—fatigue/malaise, sore throat, headache, age, and vital sign composites—provide a focused set of indicators for population-level surveillance.

*Algorithm Selection for Healthcare:* When deploying clustering in clinical settings, algorithm choice should prioritize balanced, interpretable groups (favoring K-Means or DBSCAN here) over purely statistical metrics (which favor BIRCH).

*Data Collection Priorities:* The poor discrimination of individual vital sign measurements suggests value in multi-parameter vital sign scoring systems rather than isolated readings.
