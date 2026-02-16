#import "@preview/clear-iclr:0.7.0": *

#show: clear-iclr.with(
  title: [COVID-19 Patient Clustering Analysis: A Multi-Algorithm Approach for Symptom-Based Patient Stratification],
  authors: (
    (
      names: "Authors",
      affiliation: "Institution",
      address: "Address",
      email: "email@example.com"
    ),
  ),
  keywords: ("Clustering", "COVID-19", "BIRCH", "DBSCAN", "K-Means", "Machine Learning", "Healthcare Analytics"),
  date: "February 16, 2026",
  abstract: [
    This study explores the application of clustering algorithms to identify COVID-19 patient susceptibility groups based on symptoms, vital signs, and demographic information. Using data from two hospital datasets comprising 26,237 patients, we compare the performance of BIRCH, DBSCAN, and K-Means algorithms. Initial analysis with BIRCH on the full feature set achieved a silhouette score of 0.977, while dimension reduction techniques identified fatigue/malaise, sore throat, headache, age, and a combined oxygen-fever metric as key discriminative features. Subsequent clustering on reduced dimensions yielded varied results: DBSCAN (silhouette score: 0.301, 8 clusters), K-Means (silhouette score: 0.440 with 10 clusters), and BIRCH (silhouette score: 0.977 with 3 highly imbalanced clusters). Our findings demonstrate that while high-dimensional clustering can achieve excellent separation metrics, dimension reduction with appropriate algorithm selection provides more interpretable and balanced patient stratification for clinical applications.
  ],
  accepted: none,
)

= Introduction to Clustering

The COVID-19 pandemic has placed immense pressure on hospitals, which serve as the frontline defense against the virus. Rapid and accurate identification of COVID-positive patients is crucial for managing hospital resources, ensuring patient safety, and preventing the virus's spread within healthcare facilities. Traditional methods of diagnosing COVID-19 often rely on extensive manual testing and delayed laboratory results, which can strain hospital workflows and lead to inefficient resource allocation.

This project explores the correlation of symptoms to cluster patients in order to identify susceptibility groups tailored for hospital use. By leveraging patient data such as symptoms, vital signs, and demographic information, we aim to proactively alert the population of symptoms to watch out for and support clinical decision-making.

== Algorithm Selection Rationale

Initial data visualization revealed a non-spherical, continuous manifold-like structure with possibly hierarchical or density-based patterns inside. Based on these characteristics, the following algorithms were selected:

- *BIRCH* (Balanced Iterative Reducing and Clustering using Hierarchies): Chosen for its hierarchical clustering capability, scalability, and incremental learning approach. BIRCH is particularly suitable for large datasets and can handle non-spherical clusters through its hierarchical structure.

- *DBSCAN* (Density-Based Spatial Clustering of Applications with Noise): Selected for density-based clustering after dimension reduction. DBSCAN automatically detects noise, does not require pre-specifying the number of clusters, and can identify arbitrarily shaped clusters.

- *K-Means*: Included for comparison purposes as a baseline centroid-based algorithm. While K-Means assumes spherical clusters and is sensitive to initialization, its computational efficiency and widespread use make it a valuable benchmark.

- *CURE* (Clustering Using REpresentatives): Considered for its hierarchical approach with multiple representatives per cluster but was not implemented in the final analysis.

= Dataset Discussion

== Dataset Description

The analysis utilized patient data from two hospital sources (hospital1.xlsx and hospital2.xlsx), resulting in a merged dataset with comprehensive patient information. The datasets initially contained records with varying naming conventions and languages (Turkish column names in Hospital 1), requiring substantial preprocessing.

The dataset comprises several categories of features:

*Demographics:* patient_id, admission_id, age, sex, and nationality provide basic patient information essential for population-based analysis.

*Vital Signs:* fever_temperature (°C) and oxygen_saturation (%) serve as critical physiological indicators of COVID-19 severity.

*Temporal Information:* date_of_first_symptoms and admission_date track disease progression timelines.

*Target Variable:* pcr_result indicates COVID-19 test outcome (0: negative, 1: positive).

*Symptom Features:* A comprehensive set of 25 boolean features captures patient-reported symptoms including history_of_fever, cough, sore_throat, runny_nose, wheezing, shortness_of_breath, lower_chest_wall_indrawing, chest_pain, conjunctivitis, lymphadenopathy, headache, loss_of_smell, loss_of_taste, fatigue_malaise, anorexia, altered_consciousness_confusion, muscle_aches, joint_pain, inability_to_walk, abdominal_pain, diarrhoea, vomiting_nausea, skin_rash, bleeding, and other_symptoms.

*Comorbidity Features:* 19 boolean features document pre-existing conditions such as chronic_cardiac_disease, hypertension, chronic_pulmonary_disease, asthma, chronic_kidney_disease, obesity, liver_disease, asplenia, chronic_neurological_disorder, malignant_neoplasm, chronic_hematologic_disease, AIDS_HIV, diabetes_mellitus_type_1, diabetes_mellitus_type_2, rheumatologic_disorder, dementia, tuberculosis, smoking, and other_risks.

== Data Quality Analysis

Initial exploration of both hospital datasets revealed several data quality issues requiring attention:

*Hospital 1 Issues:*
- Redundant columns: patient_id and patient_id.1 contained identical information
- Column naming inconsistencies: Turkish column names (e.g., 'basvurutarihi' for admission_date, 'gender_k=female_e=male' for sex)
- Data type mismatches: fever_temperature stored as string rather than float
- Missing values: 1,176 missing PCR results
- Inconsistent value representations: Gender encoded as 'k' (kadın/female) and 'e' (erkek/male)
- NaN values scattered across symptom columns

*Hospital 2 Issues:*
- Extensive missing data: 1,222 missing temperature values
- Categorical encoding problems: Gender field contained a third category due to data legend inconsistencies
- Column naming inconsistencies requiring standardization
- Complete rows with NaN values
- Missing symptom information

The analysis revealed systematic data collection differences between hospitals, necessitating careful harmonization strategies.

== Data Merging and Transformation

=== Column Standardization

To enable dataset integration, column names were systematically standardized:

```python
# Hospital 1 standardization
hospital1_copy.rename(columns={
    'basvurutarihi': 'admission_date',
    'patient_id.1': 'admission_id',
    'gender_k=female_e=male': 'sex'
}, inplace=True)

# Hospital 2 standardization
hospital2_copy.rename(columns={
    'country_of_residence': 'nationality'
}, inplace=True)
```

=== Dataset Integration

The two hospital datasets were concatenated row-wise to create a unified dataset:

```python
merged_hospitals = pd.concat(
    [hospital1_copy, hospital2_copy],
    axis=0,
    ignore_index=True
)
```

This merging strategy resulted in a final dataset of 26,237 patient records with harmonized feature names across all sources.

=== Feature Encoding

*Gender Encoding:* The sex variable was converted to binary encoding (Male=1, Female=0):

```python
hospital1_copy['sex'] = hospital1_copy['sex'].map({'e': 1, 'k': 0})
```

*PCR Result Encoding:* The target variable was standardized to binary format:

```python
hospital1_copy['pcr_result'] = hospital1_copy['pcr_result'].map({
    'positive': 1,
    'negative': 0
})
```

*Nationality Standardization:* Country names underwent complex normalization using ISO 3166-1 numeric codes to handle various formats and spellings:

```python
def standardize_country_name(country):
    custom_mappings = {
        "t.c.": "792",    # Turkey
        "usa": "840",     # United States
        "cyprus": "196",
        # Additional mappings...
    }

    if country in custom_mappings:
        return custom_mappings[country]

    try:
        country_obj = pycountry.countries.get(name=country) or \
            pycountry.countries.search_fuzzy(country)[0]
        return country_obj.numeric
    except (LookupError, AttributeError):
        return country

merged_hospitals["nationality_numeric"] = \
    merged_hospitals["nationality"].map(
        lambda x: standardize_country_name(x.strip().lower())
    )
```

This approach handles variations in country name formatting while maintaining numerical consistency for analysis.

== Data Cleaning

=== Missing Value Analysis and Imputation

*Temperature Data:* Trimmed mean analysis was performed to assess the impact of outliers:

```python
trimmed_mean = scstat.trim_mean(
    sorted_temp_hs1,
    proportiontocut=0.0168
)
standard_mean = sorted_temp_hs1.mean()
```

The difference between trimmed and standard means was negligible (<0.01°C), indicating outliers did not significantly skew the distribution. Temperature values were imputed using the mean:

```python
hospital1_copy['fever_temperature'].fillna(
    hospital1_copy['fever_temperature'].mean(),
    inplace=True
)
```

While temperatures such as 34.8-35.5°C (hypothermia) and 39.5-40.1°C (high fever) appeared unrealistic for typical cases, they were retained as potentially clinically significant observations.

*Oxygen Saturation:* Special consideration was given to oxygen saturation values of -1 or 0, which indicate patient death rather than measurement errors. These values were handled separately in the analysis.

*Discrete Features:* Symptom and comorbidity features were imputed using mode (most frequent value):

```python
columns_to_fill = [list of symptom and comorbidity columns]
for col in columns_to_fill:
    hospital1_copy[col].fillna(
        hospital1_copy[col].mode()[0],
        inplace=True
    )
```

=== Data Type Conversion

Boolean features were explicitly converted to integer type for computational efficiency:

```python
columns_to_convert = [symptom and comorbidity columns]
for col in columns_to_convert:
    merged_hospitals[col] = merged_hospitals[col].astype('int64')
```

=== Handling Missing Nationalities

Given that nationality plays a significant role in population density and disease spread patterns, records with missing nationality information were removed:

```python
hospital1_copy = hospital1_copy[
    ~hospital1_copy['nationality'].isnull()
]
hospital2_copy = hospital2_copy[
    ~hospital2_copy['nationality'].isnull()
]
```

== Final Dataset Characteristics

The cleaned and merged dataset comprised 26,237 patient records with the following distribution:

```
PCR Result Distribution:
- Positive (1): 22,210 patients (84.6%)
- Negative (0):  4,027 patients (15.4%)
```

This substantial class imbalance (approximately 85% positive cases) reflects the dataset's focus on COVID-positive patient populations and presents considerations for clustering algorithm interpretation.

= Initial Analysis Using BIRCH

== Data Preparation for Clustering

=== Feature Selection and Scaling

Identifier columns (patient_id, admission_id, nationality) were removed as they do not contribute to clinical pattern recognition:

```python
id_cols = ["patient_id", "admission_id", "nationality"]
df_clus = merged_hospitals.drop(columns=id_cols)
```

Given the presence of outliers in vital sign measurements, RobustScaler was selected over StandardScaler for feature normalization:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_clus)
```

RobustScaler uses the interquartile range (IQR) rather than mean and standard deviation, making it more resilient to extreme values in temperature and oxygen saturation data.

=== Initial Visualization

Principal Component Analysis (PCA) was employed to visualize the high-dimensional data in 2D space:

```python
from sklearn.decomposition import PCA

X_pca_2d = PCA(n_components=2).fit_transform(X_scaled)
```

A visualization helper function enabled consistent cluster plotting throughout the analysis:

```python
def plot_scatter(X, labels, title):
    plt.figure(figsize=(6, 5))
    if type(labels) == np.ndarray and labels.size != 0:
        unique = np.unique(labels)
        for lb in unique:
            mask = labels == lb
            plt.scatter(X[mask, 0], X[mask, 1],
                       s=12, label=f'cluster {lb}')
    else:
        plt.scatter(X[:, 0], X[:, 1],
                   cmap="tab10", s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
```

The initial visualization revealed continuous, non-spherical structure suggesting hierarchical organization.

== BIRCH Implementation

=== Initial Model with Default Parameters

BIRCH was first applied with default hyperparameters:

```python
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

birch = Birch()  # threshold=0.5, branching_factor=50, n_clusters=3
birch.fit(X_scaled)
initial_labels = birch.labels_
initial_silhouette = silhouette_score(X_scaled, initial_labels)
```

This initial configuration achieved a silhouette score of *0.9771*, indicating excellent cluster separation.

=== Hyperparameter Tuning

To potentially improve upon this strong baseline, systematic hyperparameter optimization was conducted using randomized search:

```python
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint

def compute_birch_with_hyperparams(X_data, param_distributions):
    results = []

    param_list = list(ParameterSampler(
        param_distributions,
        n_iter=50,
        random_state=21423
    ))

    for params in param_list:
        model = Birch(**params)
        labels = model.fit_predict(X_data)

        n_clusters = len(np.unique(labels))
        # Ensure meaningful clustering
        if n_clusters < 2 or n_clusters >= len(X_data):
            continue

        sil = silhouette_score(X_data, labels)

        results.append({
            **params,
            "silhouette": sil,
            "labels": labels
        })

    return results

param_distributions = {
    "threshold": uniform(0.1, 2.0),
    "branching_factor": randint(20, 100),
    "n_clusters": randint(2, 10)
}

results = compute_birch_with_hyperparams(X_scaled, param_distributions)
df = pd.DataFrame(results).sort_values("silhouette", ascending=False)
```

The hyperparameter search evaluated 50 different configurations across:
- *Threshold*: 0.1 to 2.1 (controls cluster radius)
- *Branching factor*: 20 to 100 (affects tree structure)
- *Number of clusters*: 2 to 10 (final cluster count)

=== Results

The optimization did not improve upon the initial silhouette score of 0.9771, though it produced different cluster assignments. The best configuration maintained three clusters with the following distribution:

```
Cluster 0:   503 patients (1.9%)
Cluster 1: 25,397 patients (96.8%)
Cluster 2:   337 patients (1.3%)
```

This highly skewed distribution, with one cluster containing 96.8% of patients, raised concerns about the practical utility of the clustering despite the excellent silhouette score. High silhouette scores can sometimes indicate that one cluster dominates the dataset rather than meaningful separation.

== Post-Analysis for Dimension Reduction

To improve cluster balance and interpretability, feature selection analysis was conducted to identify the most discriminative variables.

=== Feature Categorization

Features were systematically separated into boolean and continuous types:

```python
# Boolean features (symptom and comorbidity indicators)
bool_features = [
    c for c in df_clus.columns
    if df_clus[c].dropna().isin([0, 1]).all()
    and c != 'labels'
]

# Continuous features
continuous_features = list(
    set(df_clus.columns) - set(bool_features) - {"labels"}
)
```

=== Temporal Feature Analysis

Date-encoded features were examined for variability:

```python
(df_clus["admission_date"] -
 df_clus["date_of_first_symptoms"]).unique()
```

All differences equaled zero, indicating patients were admitted on the day of first symptom onset. Consequently, both date features were removed from the continuous feature set as they provided no discriminative power.

=== Continuous Variable Discrimination Analysis

Three metrics assessed the discriminative power of continuous features:

*Between-Cluster Mean Separation:* Measures how far apart cluster centers are (larger is better):

```python
desc_cont = df_clus.groupby("labels")[continuous_features].describe()
mean_sep = desc_cont.xs("mean", level=1, axis=1).std(axis=0)
```

*Within-Cluster Standard Deviation:* Measures cluster tightness (smaller is better):

```python
within_std = desc_cont.xs("std", level=1, axis=1).mean(axis=0)
```

*Discriminative Ratio:* Combines both metrics to assess overall separation quality:

```python
disc_ratio = mean_sep / within_std
```

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

```python
p = df_clus.groupby("labels")[bool_features].mean()
delta_p = p.max() - p.min()
```

*Cramér's V (Association Strength):* Quantifies statistical association between feature and cluster assignment:

```python
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    k = min(ct.shape)
    return np.sqrt(chi2 / (n * (k - 1)))

cramers_table = pd.Series({
    c: cramers_v(df_clus[c], df_clus["labels"])
    for c in bool_features
})
```

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
```python
selected_bool = ['fatigue_malaise', 'sore_throat', 'headache']
```

= Second Clustering Attempt Using Reduced Dimensions

== Dimension Reduction Strategy

Based on the post-analysis findings, a reduced feature set was constructed combining statistically and clinically significant variables:

```python
# Selected boolean symptoms plus age
id_cols = selected_bool + ["age"]
df_reduced = merged_hospitals[id_cols].copy()

# Combine oxygen_saturation and fever_temperature using PCA
reduction = merged_hospitals[
    ["oxygen_saturation", "fever_temperature"]
]
x_tmp = scaler.fit_transform(reduction)
df_reduced["oxygen_fever"] = PCA(
    n_components=1
).fit_transform(x_tmp)

# Scale the final reduced dataset
X_reduced = scaler.fit_transform(df_reduced)
```

The final reduced feature set comprised five dimensions:
- *fatigue_malaise* (boolean symptom)
- *sore_throat* (boolean symptom)
- *headache* (boolean symptom)
- *age* (continuous demographic)
- *oxygen_fever* (continuous vital sign composite)

== Visualization of Reduced Data

Three-dimensional PCA projection enabled visualization of the reduced feature space:

```python
from mpl_toolkits.mplot3d import Axes3D

def plot_scatter_3d(X, labels=None, title="3D scatter",
                    elevation=20, azim=45):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    if isinstance(labels, np.ndarray) and labels.size != 0:
        for lb in np.unique(labels):
            mask = labels == lb
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                      s=12, alpha=0.8, label=f"cluster {lb}")
        ax.legend(loc="best")
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                  s=10, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=elevation, azim=azim)
    plt.tight_layout()
    plt.show()

X_pca_3d_reduced = PCA(n_components=3).fit_transform(X_reduced)
plot_scatter_3d(X_pca_3d_reduced,
               title="Reduced Feature Space (3D PCA)")
```

== DBSCAN on Reduced Data

DBSCAN was applied to the reduced feature space with systematic hyperparameter optimization:

```python
from sklearn.cluster import DBSCAN

best_score = -1
best_params = {}
best_labels = None

eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_values = [5, 10, 15, 20]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters >= 2 and n_noise < len(labels):
            score = silhouette_score(X_reduced, labels)

            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = labels
```

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

= Conclusion

This study demonstrates the application of multiple clustering algorithms to COVID-19 patient symptom data, revealing important insights about algorithm selection and feature engineering for healthcare analytics. While BIRCH achieved excellent silhouette scores (0.977), the resulting cluster imbalance limits clinical utility. K-Means (silhouette: 0.440, 10 clusters) and DBSCAN (silhouette: 0.301, 8 clusters) on reduced feature sets provided more balanced and interpretable patient stratifications.

The dimension reduction process identified fatigue/malaise, sore throat, headache, age, and combined oxygen-fever metrics as key discriminative features. This focused feature set enables efficient symptom-based patient monitoring while maintaining clinically relevant information.

Future work should explore:
- Supervised learning approaches using PCR results as labels
- Temporal clustering tracking symptom evolution over admission duration
- Integration of comorbidity features for risk-adjusted stratification
- Validation on additional hospital datasets to assess generalizability
- Investigation of the severely imbalanced underlying data distribution

The findings underscore that effective healthcare clustering requires balancing statistical performance with clinical interpretability, domain knowledge integration, and practical deployment considerations.
