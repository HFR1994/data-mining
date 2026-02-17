= Data Analysis 

== Initial Cluster Interpretation

Analysis of the initial BIRCH clustering with three clusters revealed distinct patient profiles:

*Cluster 0 (503 patients, 1.9%):* Younger patients (mean age: 38.5 years) with milder symptoms—lower fever (37.3°C), and lower rates of fever history (22.5%) and cough (15.9%). Are mostly healthy.

*Cluster 1 (25,397 patients, 96.8%):* The dominant cluster with mean age 43.1 years, have moderate symptoms including 51% fever history and 29.6% cough rate. This means that 43.1 year old patients are more prune to disease

*Cluster 2 (337 patients, 1.3%):* Youngest group (mean age: 34.9 years) with mixed symptom presentation—41.2% fever history, 24.3% cough, and 25.8% sore throat. May represent a distinct symptomatic subgroup. Factors such street exposure could be accredited for.

#figure(
  caption: [Heatmap table of mean statistics Initial],
  image("../assets/output_birch.png", width: 90%),
)

== Final Cluster Interpretation

Rather they're analyzing the cluster individually, label by label we can break it down by the in the mean and color code it to make it easier to breakdown and sort it based on higher variance statistics.

#figure(
  caption: [Heatmap Table of mean statistics Part1],
  image("../assets/output1.png", width: 90%),
)

#v(0.3cm)

#figure(
  caption: [Heatmap Table of mean statistics Part2],
  image("../assets/output2.png", width: 90%),
)

#v(0.3cm)

#figure(
  caption: [Heatmap Table of mean statistics Part3],
  image("../assets/output3.png", width: 90%),
)

#v(0.3cm)

Based on our previous research, we now can assign cluster labels by age groups since they groupable means with similar standard deviation.

Judging by the cluster labels, Groups 2 and 7 corresponding to a certain subset of Young Adults and Older Adults share the same disease map (cough, hypertension, vomiting nausea, shortness of breath, diarrhea and runny nose). This is coherent with the initial BIRCH analysis since it detect that groups after 43 are more to diseases. It also states that there is a of 30 year old people that have the same problems.

Another interesting result is that non smoking patients are more prone to bleeding which is rectified by #underline[#link("https://pubmed.ncbi.nlm.nih.gov/30597498/")[this NIH article].]

Since initial data set shows a PCR result rather the output of a series of diseases we can assume this is done by exposure. It is also interesting to show the 30 year old group is more prone to fatigue malaise. Apparently, there seems to be no correlation and smoking with any disease. In order to properly correlate values we need to have access to the physical lifestyle of patients to match them properly.