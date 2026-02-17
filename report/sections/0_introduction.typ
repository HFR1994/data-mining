= Introduction to Proyect

The COVID-19 pandemic has placed immense pressure on hospitals, which serve as the frontline defense against the virus. Rapid and accurate identification of COVID-positive patients is crucial for managing hospital resources, ensuring patient safety, and preventing the virus's spread within healthcare facilities.
Traditional methods of diagnosing COVID-19 often rely on extensive manual testing and delayed laboratory results, which can strain hospital workflows and lead to inefficient resource allocation.

Although this project is based on COVID-19 datasets, it seeks to explore the correlation of symptoms in clustered patients to identify 
susceptibility symptoms based on cleared segmentations. By leveraging patient data such as symptoms, vital signs, and demographic information, 
I aim to proactively correlate certain group of symptoms to explain data.

== Approach and rational

In order to achieve this correlation, clustering methods where used as a technique to separate data. Initial data visualization revealed 
a non-spherical, continuous cascade like structure with density-based patterns. Based on these characteristics an appropriate clustering method was used to 
attempt to organize groups so we could later breakdown characteristics and look for the most promising fields. As an initial approach BURCH was used 
for its hierarchical clustering capability, scalability, and incremental learning approach. Since is particularly suitable for large datasets, it can handle non-spherical clusters with no apparent structure.