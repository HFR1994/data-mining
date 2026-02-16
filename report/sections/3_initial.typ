= Initial Analysis Using BIRCH

== Data Preparation for Clustering

=== Feature Selection and Scaling

As a primary analysis, we want to keep the columns that represented A significant difference rather than just a random assignment. Given the current column distribution columns such patient_id, admission_id, nationality were removed as they do not contribute to clinical pattern recognition.

Furthermore, In order to avoid outliers in vital sign measurements, RobustScaler was selected as a normalization. RobustScaler ws used since it considers interquartile range (IQR) rather than mean and standard deviation. This makes it more resilient to extreme values in temperature and oxygen saturation data.

=== Initial Visualization

Before starting to cluster, Principal Component Analysis (PCA) was employed to visualize the high-dimensional data in 2D space based on the remaining data.

#figure(
  caption: [Initial Cluster visualization],
  image("../assets/initial.png", width: 80%),
)

My initial thoughts revealed that this data was highly fragmented and even with a certain type of hierarchy it was clear that it could be partition into multiple clusters, as certain divisions were clearly visible.

Since the mage revealed continuous, non-spherical structure this left out certain clusters such as KMeans from the equation.

== BIRCH Implementation

Given the high density form in the representation, BIRCH was selected as a mechanism for initial clustering. The main purpose of this implementation is to identify which are the most influential Given that a proper cluster must have between five to seven columns. 

=== Initial Model with Default Parameters

In order to have a proper baseline BIRCH was first applied with default hyper-parameters $("threshold" = 0.5, "branching_factor" = 50, "n_clusters" = 3)$. This yield a silhouette score of *0.9771*, indicating excellent cluster separation with a proper representation:

#figure(
  caption: [Birch Initial cluster],
  image("../assets/birch_initial.png", width: 80%),
)

=== Hyperparameter Tuning

To potentially improve upon this strong baseline, systematic hyper-parameter was conducted using randomized search over parameter space where:

- $"threshold" ~ "U"(0.1, 2.0)$
- $"branching_factor" in(2, 100)$
- $"n_clusters" in(2, 10)$
- $n = 50$ iterations

=== Results

The optimization did not improve upon the initial silhouette score of 0.9771, though it produced different cluster assignments. The best configuration maintained three clusters with the following distribution:

```
Cluster 0:   503 patients (1.9%)
Cluster 1: 25,397 patients (96.8%)
Cluster 2:   337 patients (1.3%)
```

This representation show a highly skewed distribution. One cluster contains 96.8% of patients, raised concerns about the practical utility of the clustering despite the excellent silhouette score. High silhouette scores can sometimes indicate that one cluster dominates the dataset rather than meaningful separation.

From what we learned in the previous analysis the dataset is imbalanced which can lead improper or highly group data points.

#figure(
  caption: [Birch 2D representation of the cluster],
  image("../assets/birch_final.png", width: 70%),
)

This can be seen better from a 3D perspective view: 

#figure(
  caption: [Birch 3D representation of thecluster],
  image("../assets/birch_initial_3d.png", width: 70%),
)