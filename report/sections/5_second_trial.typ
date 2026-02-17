#show figure.where(
  kind: table
): set figure.caption(position: bottom)

= Second Clustering Attempt Using Reduced Dimensions

The 3D reduction can be classified as dispersed and highly concentrated regardless of the axis. This is where density based algorithms operate better. 

== DBScan on Reduced Data

Based on class notes, DBSCAN it's a better alternative since it Has native support for both density based clustering and doesn´t need a $K$ value to be specified. The implementation was executed with hyper-parameter optimization in mind.

Where:

- $epsilon in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}$ (neighborhood radius)
- $m in {5, 10, 15, 20}$ (minimum points)

Given that $K ≥ 2$ and $N_i < "labels"$, where $N_i$ are noise points.

=== Results

After 24 runs the optimal DBSCAN configuration achieved was:

- *Best parameters:* eps=0.5, min_samples=5
- *Silhouette score:* 0.301 (moderate separation)
- *Number of clusters:* 8
- *Noise points:* 0

This means that DBSCAN successfully identified eight distinct patient groups without classifying any samples as noise. The moderate silhouette score (0.301) indicates overlapping but distinguishable clusters, suggesting genuine structure in the reduced feature space. Unlike BIRCH on full features, DBSCAN produced more balanced cluster sizes, enhancing practical interpretability.

If we plot it, we get the following result:

#figure(
  caption: [DBScan Reduced Feature Set 2D visualization],
  image("../assets/dbscan_2d.png", width: 80%),
)

#v(0.5cm)

However, there is still a clear imbalance between clusters. if we count the values we get the following ratio:

```python
Label 0: 13136 samples (50.15%)
Label 1: 1763  samples (6.73%)
Label 2: 2865  samples (10.94%)
Label 3: 3680  samples (14.05%)
Label 4: 383   samples (1.46%)
Label 5: 489   samples (1.87%)
Label 6: 3145  samples (12.01%)
Label 7: 734   samples (2.80%)
```

At first glance this might seem an error. Given the proximity of some lines, 1st instinct could dictate to reduce the number of clusters. However, If we plot 3D representation we can clearly see a separation:

#figure(
  caption: [DBScan Reduced Feature Set 3D visualization],
  image("../assets/dbscan_3d.png", width: 80%),
)

#v(0.3cm)

Clusters don't segment only in the X and Y axis. If we take in consideration the Z, separations become more clear. Also as denotated by the ratio, some clusters have a imbalance of values. This means that we need to further divide them, or find another clustering technique.

== K-Means on Reduced Data

Taking into consideration the previous approach, I thought about rectifying my previous assumption and each line as a centric based cluster in three dimensions. Based on this, K-Means clustering seemed like a feasible approach to establish a cluster.

Using the same approach as DBScan, cluster was trained using hyper-parameters iterating over number of clusters (from 2 to 10).

=== Results

For each cluster the following silhouette scores was computed:

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

As noticed the optimal K-Means was 10 clusters:
- *Best k:* 10 clusters
- *Silhouette score:* 0.440 (moderate-good separation)

If we plot it, we get a completely different result and interpretations from previous clustering methods:

#figure(
  caption: [KMeans Reduced Feature Set 2D visualization],
  image("../assets/kmeans_2d.png", width: 80%),
)

#v(0.3cm)

If we analyze the image above, we can a alternative method being offered, were clusters are in the y access rather than the X axis. At first glance this might seen inlogical. However, remembering there is an imbalance especially in the first diagonal line, it seems a good idea to help it break-it up.

If we breakdown the clusters and count the values, we get the following ratio:

```python
Label 0: 4861 samples (18.56%)
Label 1: 3464 samples (13.22%)
Label 2: 2245 samples (8.57%)
Label 3: 1358 samples (5.18%)
Label 4: 2727 samples (10.41%)
Label 5: 2244 samples (8.57%)
Label 6: 4507 samples (17.21%)
Label 7: 1897 samples (7.24%)
Label 8: 1621 samples (6.19%)
Label 9: 1271 samples (4.85%)
```

As notice, the half aligned much better with this implementation, since they are more balanced. Furthermore, if we plot it in 3D it starts to make more sense:

#figure(
  caption: [KMeans Reduced Feature Set 2D visualization],
  image("../assets/kmeans_3d.png", width: 80%),
)

#v(0.3cm)

Surprisingly enough, K-Means demonstrated better assignment, especially with higher k values. Ranking with the highest silhouette score at k=10. This result outperformed DBSCAN (0.301) on the reduced feature set, suggesting that spherical cluster assumptions reasonably approximate the data structure after dimension reduction.

== BIRCH on Reduced Data

Finally just to compare results, BIRCH was re-applied to the reduced feature space with the same hyper-parameter optimization:

- $"threshold" ~ "U"(0.1, 2.0)$
- $"branching_factor" in(2, 100)$
- $"n_clusters" in(2, 10)$
- $n = 50$ iterations

=== BIRCH Reduced Results

After each computation, BIRCH achieved a very high silhouette score, even better then KMeans:

- *Silhouette score:* 0.977 (excellent, matching full-feature performance)
- *Number of clusters:* 3

If we plot the cluster, we see the following result:

#figure(
  caption: [Burch Reduced Feature Set 2D visualization],
  image("../assets/birch_2d_reduced.png", width: 80%),
)

#v(0.3cm)

Despite having a high silhouette score, we can see that is not a definitive measurement to determine proper clustering. From the three different types of clusters Burch was the one who operated more poorly. 

Furthermore, if we get the ratio of samples versus clusters it shows a great imbalance: 

```python
Label 0: 503   samples (1.92%)
Label 1: 25355 samples (96.79%)
Label 2: 337   samples (1.29%)
```

Even though BIRCH maintained a exceptionally high silhouette score even after dimension reduction, the cluster distribution remained severely imbalanced. The near-identical performance on both full and reduced feature sets suggests BIRCH is primarily identifying the same dominant patient subgroup (96.8% in Cluster 1) regardless of feature space dimensionality.

If we take a look add the 3D plot this even becomes more evident:

#figure(
  caption: [Burch Reduced Feature Set 3D visualization],
  image("../assets/birch_3d_reduced.png", width: 80%),
)

#v(0.3cm)

In this representation, cluster _0_ and _2_ Are completely absorbed by cluster _1_.

= Results and Discussion

== Algorithm Performance Comparison

The clustering experiments yielded contrasting results across algorithms and feature spaces:

#v(0.5cm)

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

#v(0.3cm)

== Feature Importance Findings

After process, we group the clustered based on its corresponding label and certain pattern start to emerge:

*Continuous Features:*
- *Age:* Emerged as the most discriminative continuous variable (ratio: 0.28), indicating partial cluster separation by patient age groups. Even after clustering, patterns are still are visible:

  #v(0.3cm)

  #figure(
    table(
      columns: (auto, auto, auto),
      align: (left, left, left),
      inset: 6pt,
      stroke: (x: 0.5pt, y: 0.5pt),
    
      [*Label*], [*Age Mean*], [*Age Std*],
    
      [0], [31.04], [5.02],
      [1], [71.27], [8.00],
      [2], [31.27], [8.51],
      [3], [66.32], [10.46],
      [4], [33.73], [8.48],
      [5], [32.18], [8.47],
      [6], [48.86], [5.83],
      [7], [59.94], [9.51],
      [8], [32.54], [8.75],
      [9], [13.31], [6.01],
    ),
    caption: [Age separation by KMeans Cluster labels]
  )
  
  At a first glance, it clear data cluster can be classified as followed:

  #v(0.3cm)

  #figure(
    table(
      columns: (auto, auto, auto),
      align: (center, center, left),
      inset: 6pt,
      stroke: (x: 0.5pt, y: 0.5pt),
    
      [*Cluster*], [*Avg age*], [*Interpretation*],
    
      [9], [~13], [Children],
      [0, 2, 4, 5, 8], [~31–34], [Young adults],
      [6], [~49], [Middle-aged],
      [7], [~60], [Older adults],
      [1, 3], [~66–71], [Elderly],
    ),
    caption: [Age groups by KMeans Cluster labels]
  )


== Methodological Insights

*Silhouette Score Limitations:* High silhouette scores (0.977 for BIRCH) do not guarantee clinically useful clustering. The severely imbalanced distribution suggests the metric captured one dominant group's homogeneity rather than meaningful patient stratification.

*Algorithm-Data Interaction:* BIRCH's hierarchical structure may be overly sensitive to the dataset's inherent imbalance. DBSCAN and K-Means, operating on density and centroid principles respectively, produced more balanced groupings on reduced features.

