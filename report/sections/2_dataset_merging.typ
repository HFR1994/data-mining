#import "@preview/tabbyterms:0.1.0" as tabbyterms: terms-table

= Data Merging and Transformation

== Column Standardization

To enable dataset integration, column names were systematically standardized:

*Hospital 1* column mappings:

#align(center)[
  #show: tabbyterms.style.default-styles
  #terms-table[
    / basvurutarihi: admission_date
    / patient_id.1: admission_id
    / gender_k=female_e=male: sex
  ]
]
*Hospital 2* column mappings:

#align(center)[
  #show: tabbyterms.style.default-styles
  #terms-table[
    / country_of_residence: nationality
  ]
]

== Dataset cleaning

=== Handling Missing Nationalities

Given that nationality plays a significant role in population density and disease spread patterns, missing nationality were removed using pandas filtering function.

=== Feature Encoding

*Gender Encoding:* The _sex_ variable was encoded to binary: 

#align(center)[
  #show: tabbyterms.style.default-styles
  #terms-table[
    / Male: 0
    / Female: 1
  ]
]

*PCR Result Encoding:* The _pcr_result_ variable was standardized to binary format:

#align(center)[
  #show: tabbyterms.style.default-styles
  #terms-table[
    / positive: 1
    / negative: 0
  ]
]

*Nationality Standardization:* Country names underwent complex normalization using ISO 3166-1 numeric codes to handle various formats and spellings using a custom mapping table:

```python
custom_mappings = {
        "t.c.": "792",  # Numeric code for Turkey,
        "kuzey kibris tãœrk cum.": "792",  # Turkey again
        "diäÿer": "792",  # Turkish word
        "united kingdom (great britain)": "826",
        "usa": "840",  # Numeric code for United States
        "iran (islamic republic of iran)": "364",
        ....
}
```

This approach handles variations in country name formatting while maintaining numerical consistency for analysis.

== Data Cleaning

=== Missing Value Analysis and Imputation

After checking missing values of both dataset a strategy was implemented to replace null values based on _mode and mean values_.

*Temperature Data:* At first glance, in order to avoid outliers  trimmed mean was used. However, the difference between trimmed and standard means was negligible "$(<0.01°C)$", indicating outliers did not significantly skew the distribution. So, temperature values were imputed using the standard mean.

While temperatures such as 34.8-35.5°C (hypothermia) and 39.5-40.1°C (high fever) appeared unrealistic for typical cases, they were retained as potentially clinically significant observations.

*Oxygen Saturation:* After analysis, abnormal values such as -1 or 0 where found. This indicated patient death rather than measurement errors. These values were handled separately in the analysis by dropping them

*Age:* After comparing the range of values, It was clear that certain ages we're not possible. The initial analysis show some rows that had "-1" values. As a result, this rows were dropped.

*Discrete Features:* Null values for comorbidity features were imputed using mode (most frequent value)

```python
columns_to_fill_mode = [
'history_of_fever', 'other_symptoms', 'bleeding', 'sex', 'chronic_kidney_disease', 'obesity', 'liver_disease',
'asplenia', 'chronic_neurological_disorder', 'chronic_hematologic_disease','aids_hiv', 'diabetes_mellitus_type_1', 'diabetes_mellitus_type_2', 'rheumatologic_disorder', 'dementia'
]
```

=== Data Type Conversion

After classifying a list of boolean values the following list of columns were explicitly converted to integer type for computational efficiency.

```python
columns_to_convert = [
  
'history_of_fever', 'cough', 'sore_throat', 'runny_nose',
'wheezing', 'shortness_of_breath', 'lower_chest_wall_indrawing', 'chest_pain','conjunctivitis', 'lymphadenopathy', 'headache', 'loss_of_smell', 'loss_of_taste','fatigue_malaise', 'anorexia', 'altered_consciousness_confusion', 'muscle_aches', 'joint_pain', 'inability_to_walk', 'abdominal_pain', 'diarrhoea', 'vomiting_nausea', 'skin_rash', 'bleeding', 'other_symptoms', 'chronic_cardiac_disease', 'hypertension', 'chronic_pulmonary_disease', 'asthma', 'chronic_kidney_disease', 'obesity','liver_disease', 'asplenia', 'chronic_neurological_disorder', 'malignant_neoplasm',     'chronic_hematologic_disease', 'AIDS_HIV', 'diabetes_mellitus_type_1', 'diabetes_mellitus_type_2', 'rheumatologic_disorder', 'dementia', 'tuberculosis','smoking', 'other_risks', 'age', 'admission_id', 'aids_hiv', 'sex', 'pcr_result'

]
```

== Dataset Integration

Finally, the two hospital datasets were concatenated row-wise to create a unified dataset where rows were concatenated with reset indices. This merging strategy resulted in a final dataset of 26,237 patient records with harmonized feature names across all sources.