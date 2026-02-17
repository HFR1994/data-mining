= Conclusion

This study demonstrates the application of multiple clustering algorithms to COVID-19 patient symptom data, revealing important insights even outside the COVID-19 scope. Multiple clustering methods can be use to achieve proper clustering. Methods like, BIRCH have excellent silhouette scores (0.977) but results in cluster imbalance. However, it was proven to be useful to find potential candidate features to be extracted. Clusters like K-Means (silhouette: 0.440, 10 clusters) and DBSCAN (silhouette: 0.301, 8 clusters) on reduced feature sets have proven to  more balanced and interpretable patient stratifications. These proves the theory, as clustering methods should have between five to seven features.

Future work should explore:
- Supervised learning approaches as labels to correlate results
- Integration of comorbidity features for risk-adjusted stratification
- Investigation of the severely imbalanced underlying data distribution
- Proper background history to relate lifestyle to features

== Clinical Implications

The clustering results offer several potential applications patient management:

*Symptom Monitoring:* The key discriminative are fatigue_malaise, sore throat, headache, age, and a reduction between oxygen and fever.

*Algorithm Selection for Healthcare:* Results are heavily biassed do the imbalance of results. Further efforts should prioritize balanced and interpretable groups.
