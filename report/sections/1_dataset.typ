= Dataset Discussion

== Dataset Description

The analysis utilized patient data from two hospital sources (hospital1.xlsx and hospital2.xlsx) which stay anonymous for health concerns. 
The datasets initially showed that records had varying naming conventions and languages (ej: Turkish column names in Hospital 1).
Before attempting to merge dataset into a single standarized file we need to analyze it individually

=== Hospital 1

With a total of 54 initial columns, the dataset is organized into several distinct categories to provide a complete view of patient health and disease progression. 
The dataset includes demographics such as patient_id, age, sex, and nationality which can be used to establish a baseline for 
population based analysis.

Temporal information is also present with information such as date of first symptoms and admission tracks which let´s us remove the temporal format by subtracting the dates 
to a clear atemporal metric. Furthermore, the dataset contains both continious and boolean based values. The dataset is partioned into:

```python
Data columns (total 54 columns):
 #   Column                           Non-Null Count  Dtype         
---  ------                           --------------  -----         
 0   patient ID                       14712 non-null  int64         
 1   patient ID.1                     14712 non-null  int64         
 2   nationality                      14712 non-null  object        
 3   age                              14712 non-null  int64         
 4   gender K=female E=male           14712 non-null  object        
 5   date_of_first_symptoms           14712 non-null  datetime64[ns]
 6   BASVURUTARIHI                    14712 non-null  datetime64[ns]
 7   fever_temperature                14244 non-null  float64       
 8   oxygen_saturation                14708 non-null  float64       
 9   history_of_fever                 14712 non-null  int64         
 10  cough                            14712 non-null  int64         
 11  sore_throat                      14712 non-null  int64         
 12  runny_nose                       14712 non-null  int64         
 13  wheezing                         14712 non-null  int64         
 14  shortness_of_breath              14712 non-null  int64         
 15  lower_chest_wall_indrawing       14712 non-null  int64         
 16  chest_pain                       14712 non-null  int64         
 17  conjunctivitis                   14712 non-null  int64         
 18  lymphadenopathy                  14712 non-null  int64         
 19  headache                         14712 non-null  int64         
 20  loss_of_smell                    14712 non-null  int64         
 21  loss_of_taste                    14712 non-null  int64         
 22  fatigue_malaise                  14712 non-null  int64         
 23  anorexia                         14712 non-null  int64         
 24  altered_consciousness_confusion  14712 non-null  int64         
 25  muscle_aches                     14712 non-null  int64         
 26  joint_pain                       14712 non-null  int64         
 27  inability_to_walk                14712 non-null  int64         
 28  abdominal_pain                   14712 non-null  int64         
 29  diarrhoea                        14712 non-null  int64         
 30  vomiting_nausea                  14712 non-null  int64         
 31  skin_rash                        14712 non-null  int64         
 32  bleeding                         14712 non-null  int64         
 33  other_symptoms                   14712 non-null  int64         
 34  chronic_cardiac_disease          14712 non-null  int64         
 35  hypertension                     14712 non-null  int64         
 36  chronic_pulmonary_disease        14712 non-null  int64         
 37  asthma                           14712 non-null  int64         
 38  chronic_kidney_disease           14705 non-null  float64       
 39  obesity                          14690 non-null  float64       
 40  liver_disease                    14706 non-null  float64       
 41  asplenia                         14690 non-null  float64       
 42  chronic_neurological_disorder    14710 non-null  float64       
 43  malignant_neoplasm               14712 non-null  int64         
 44  chronic_hematologic_disease      14710 non-null  float64       
 45  AIDS_HIV                         14710 non-null  float64       
 46  diabetes_mellitus_type_1         14709 non-null  float64       
 47  diabetes_mellitus_type_2         14710 non-null  float64       
 48  rheumatologic_disorder           14710 non-null  float64       
 49  dementia                         14710 non-null  float64       
 50  tuberculosis                     14712 non-null  int64         
 51  smoking                          14712 non-null  int64         
 52  other_risks                      14712 non-null  int64         
 53  PCR_result                       13536 non-null  object        
dtypes: datetime64[ns](2), float64(13), int64(36), object(3)
memory usage: 6.1+ MB
```

Which by a simple naked eye count of column statistics we can deduce that:

- 6 columns are discreate
- 3 columns are continous values
- 45 columns are categorical (boolean) variables 

However, data encompassing everything from respiratory distress to 19 comorbidity features which include pre-existing conditions like hypertension, diabetes, and chronic pulmonary disease.

A simple correlation matrix shows that most of values don´t have linear dependencies between each other, except for a handfull like `cought`, `history_of_fever`, `sore_thorugh`, `runny_nose` which have a big correlation value.

#figure(
  caption: [Hospital A Correlation Matrix.],
  image("../assets/correlation_a.png", width: 80%)
)

#v(0.3cm)

This is expected, as this symptoms usually group despite the disease. Furthermore, the correlation matrix confirmed that patient ID and patient ID.1 are identical, allowing for the removal of redundant identifiers. 
Data types need to be standarized as there are some categorical values expressed as floats. 

Since the dataset is missing values rows in columns such a `pcr_results`, this are expressed as objects which allows to express a total of 1,176 null values.
Data needs to also undergo transformation since is encoded using the following legend (0: negative, 1: positive, 2: no result) and (K: make and E: female) for the sex column. 

Lastly, Turkish labels needs to be changed to english. Fortunately, no duplicate rows were found. As I final fought, I think that the high variation in critical traits like fever temperatures suggests that can do dimensionality reduction improve models performance and focus on the most relevant data points.

=== Hospital 2

On the contrary, Hospital 2 also includes a total of 54 initial columns, the dataset is also organized into several distinct categories to provide. 
Demographics such as patient_id, age, sex, and country_of_residence can be directly mapped to Hospital 1 after proper standarization.

We have the same temporal information and most of the column names are present in both datasets even though the layout is different.

The dataset contains the both continious and boolean expressed as float numbers:

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 12737 entries, 0 to 12736
Data columns (total 54 columns):
 #   Column                           Non-Null Count  Dtype         
---  ------                           --------------  -----         
 0   patient_id                       12734 non-null  float64       
 1   admission_id                     12734 non-null  float64       
 2   country_of_residence             12734 non-null  object        
 3   age                              12734 non-null  float64       
 4   sex                              12734 non-null  object        
 5   date_of_first_symptoms           12734 non-null  datetime64[ns]
 6   admission_date                   12734 non-null  datetime64[ns]
 7   fever_temperature                11515 non-null  float64       
 8   oxygen_saturation                12730 non-null  float64       
 9   history_of_fever                 12729 non-null  float64       
 10  cough                            12736 non-null  float64       
 11  sore_throat                      12736 non-null  float64       
 12  runny_nose                       12734 non-null  float64       
 13  wheezing                         12734 non-null  float64       
 14  shortness_of_breath              12734 non-null  float64       
 15  lower_chest_wall_indrawing       12734 non-null  float64       
 16  chest_pain                       12734 non-null  float64       
 17  conjunctivitis                   12734 non-null  float64       
 18  lymphadenopathy                  12734 non-null  float64       
 19  headache                         12736 non-null  float64       
 20  loss_of_smell                    12736 non-null  float64       
 21  loss_of_taste                    12736 non-null  float64       
 22  fatigue_malaise                  12736 non-null  float64       
 23  anorexia                         12736 non-null  float64       
 24  altered_consciousness_confusion  12736 non-null  float64       
 25  muscle_aches                     12736 non-null  float64       
 26  joint_pain                       12736 non-null  float64       
 27  inability_to_walk                12736 non-null  float64       
 28  abdominal_pain                   12736 non-null  float64       
 29  diarrhoea                        12736 non-null  float64       
 30  vomiting_nausea                  12736 non-null  float64       
 31  skin_rash                        12736 non-null  float64       
 32  bleeding                         12698 non-null  float64       
 33  other_symptoms                   12698 non-null  float64       
 34  chronic_cardiac_disease          12736 non-null  float64       
 35  hypertension                     12736 non-null  float64       
 36  chronic_pulmonary_disease        12736 non-null  float64       
 37  asthma                           12736 non-null  float64       
 38  chronic_kidney_disease           12736 non-null  float64       
 39  obesity                          12737 non-null  int64         
 40  liver_disease                    12737 non-null  int64         
 41  asplenia                         12737 non-null  int64         
 42  chronic_neurological_disorder    12737 non-null  int64         
 43  malignant_neoplasm               12737 non-null  int64         
 44  chronic_hematologic_disease      12737 non-null  int64         
 45  AIDS_HIV                         12737 non-null  int64         
 46  diabetes_mellitus_type_1         12737 non-null  int64         
 47  diabetes_mellitus_type_2         12737 non-null  int64         
 48  rheumatologic_disorder           12737 non-null  int64         
 49  dementia                         12737 non-null  int64         
 50  tuberculosis                     12737 non-null  int64         
 51  smoking                          12737 non-null  int64         
 52  other_risks                      12737 non-null  int64         
 53  PCR_result                       12703 non-null  object        
dtypes: datetime64[ns](2), float64(35), int64(14), object(3)
memory usage: 5.2+ MB
```

When trying to group columns with an eye count, I got the same result as last time:

- 6 columns are discrete
- 3 columns are continuos values
- 45 columns are boolean variables 

However, they don´t have the same discrete columns between both datasets. The correlation matrix, shows a different much less linear dependency between values:

#figure(
  caption: [Hospital B Correlation Matrix.],
  image("../assets/correlation_b.png", width: 80%),
)

#v(0.3cm)

Being the two highest `fatigue_malaise` and `muscle_aches` with a 40% correlation which is still a low value. Finally, the dataset faces several structural challenges, most notably with a high volume of missing information, including 1,222 missing entries for temperature and various null values across multiple columns.
Finally, data integrity is compromised by the presence of entire rows containing only NaN values.

The dataset shows the same misclassification discrepancy in sex (K: make and E: female) with row imbalance and and additional 3rd value that was poorly parsed. To ensure a reliable analysis, the column names must be standardized and quality issues must be addressed through rigorous cleaning and preprocessing.