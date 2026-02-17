#import "@preview/clear-iclr:0.7.0": iclr2025

#let authors = (
  (
    names: ([Hector Carlos Flores Reynoso],),
    affilation: [
      Department of Computer Science \
      Università di Trento
    ],
    address: [Trento, Italy],
    email: "hector.floresreynoso@studenti.unitn.it",
  ),
)

#show: iclr2025.with(
  title: [COVID-19 Patient Clustering Analysis],
  authors: authors,
  keywords: ("Clustering", "COVID-19", "BIRCH", "DBSCAN", "K-Means", "Machine Learning", "Healthcare Analytics"),
  abstract: [
    This study explores the application of clustering algorithms to identify COVID-19 patient susceptibility groups based on symptoms, vital signs, and demographic information. Using data from two hospital datasets comprising 26,237 patients, we compare the performance of BIRCH, DBSCAN, and K-Means algorithms. Initial analysis with BIRCH on the full feature set achieved a silhouette score of 0.977, while dimension reduction techniques identified fatigue/malaise, sore throat, headache, age, and a combined oxygen-fever metric as key discriminative features. Subsequent clustering on reduced dimensions yielded varied results: DBSCAN (silhouette score: 0.301, 8 clusters), K-Means (silhouette score: 0.440 with 10 clusters), and BIRCH (silhouette score: 0.977 with 3 highly imbalanced clusters). Our findings demonstrate that while high-dimensional clustering can achieve excellent separation metrics, dimension reduction with appropriate algorithm selection provides more interpretable and balanced patient stratification for clinical applications.
  ],
  bibliography: none,
  appendix: none,
  accepted: none,
)

#include "sections/0_introduction.typ"
#include "sections/1_dataset.typ"
#include "sections/2_dataset_merging.typ"
#include "sections/3_initial.typ"
#include "sections/4_post_analysis.typ"
#include "sections/5_second_trial.typ"
#include "sections/6_analysis.typ"
#include "sections/7_conclusions.typ"
