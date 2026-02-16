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
