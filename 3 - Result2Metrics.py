# Databricks notebook source
# MAGIC %md
# MAGIC # Result ==> Metric

# COMMAND ----------

import wntr
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, lit
# Set colormap for network maps
cmap=plt.cm.get_cmap('YlOrRd')

# COMMAND ----------

# copy inp file from mount to drive
inp_file_name = "HanoiOptimized.inp"
dbutils.fs.cp("dbfs:/mnt/dlwadlsgen2/waterlink/99QuantumSensorPlacementDEL20/"+inp_file_name, "file:/databricks/driver/"+inp_file_name)
# Create water network model 
wn = wntr.network.WaterNetworkModel(inp_file_name)
wn_dict = wn.to_dict()

# Some graph calculations, to allow for future metrics
G = wn.get_graph()
uG = G.to_undirected()
import networkx as nx 
topological_distance=dict(nx.shortest_path_length(uG))
## topological_distance['1']['30']  # example how to derive the distance between 2 nodes

# COMMAND ----------

# Read simulation data
query = """SELECT * FROM delta.`dbfs:/mnt/dlwadlsgen2/waterlink/99QuantumSensorPlacementDEL20/simulation_results`"""
sdf_simulation_results = spark.sql(query).withColumn("LeakDemand",col("LeakDemand").cast("double")).withColumn("PressureDrop",col("PressureDrop").cast("double"))
pdf_simulation_results = sdf_simulation_results.toPandas().set_index(['LeakDemand', 'LeakNode', 'SensorLocation']).unstack('SensorLocation')
pdf_simulation_results.columns = pdf_simulation_results.columns.droplevel()
pdf_simulation_results = pdf_simulation_results.reset_index()
pdf_simulation_results.pop('1')   #watertank solution for now
pdf_simulation_results_cos = pdf_simulation_results.copy()
X_columns = pdf_simulation_results.columns.to_list()
X_columns.remove('LeakDemand')
X_columns.remove('LeakNode')

# COMMAND ----------

pdf_simulation_results['denominator'] = np.sqrt(pdf_simulation_results[X_columns].pow(2).sum(axis=1))

# COMMAND ----------

for col in X_columns:
  pdf_simulation_results_cos[col] = pdf_simulation_results[col]/pdf_simulation_results['denominator']
pdf_simulation_results_cos = pdf_simulation_results_cos.dropna()

# COMMAND ----------

sensor_set = ['11', '15', '25'] ## best solution ==> (0.08, 0.26, 0.4032258064516129, 0.16580645161290322)
                                        ## ATD       (4.976, 2.212, 1.3489159891598916, 2.6417611741160774)

                                               ## cosine-data
                                                            ## Accuracy : (0.12354463130659767, 0.1649417852522639, 0.703104786545925, 0.5194049159120311)
                                                            ## ATD :      (3.5254010695187166, 2.556970509383378, 0.7159244264507423, 1.1245650661099513)
  
## sensor_set = ['12', '15', '25'] ## 10-th solution ==> (0.08580645161290322, 0.28193548387096773, 0.4341935483870968, 0.17548387096774193)

                                        ## ATD                          (3.352, 2.0657276995305165, 1.2401360544217688, 2.558666666666667)

## sensor_set = ['12', '21', '27'] ## Article Santos-Ruiz e.a. ==> (0.08580645161290322, 0.3161290322580645, 0.5019354838709678, 0.1896774193548387)
                                                            ## ATD : (4.617333333333334, 1.968371467025572, 1.1338742393509127, 2.52320107599193)
  
                                              ## cosine-data
                                                            ## Accuracy : (0.1203104786545925, 0.17529107373868047, 0.7179818887451488, 0.47283311772315656)
                                                            ## ATD : (3.514705882352941, 2.791695030633084, 0.6292813969106783, 1.5053475935828877)
      
      
      
## sensor_set = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '4', '5', '6', '7', '8', '9']
                                   ## All sensors ==> supposed to deliver the best possible solution
                                                    ## Accuracy : (0.08967741935483871, 0.7677419354838709, 0.7819354838709678,  0.3096774193548387)
                                                    ## ATD      : (4.854666666666667,   0.5814266487213997, 0.32454361054766734, 2.0981963927855714)
                                              ## cosine-data
                                                            ## Accuracy : (0.09702457956015524, 0.6798188874514877, 0.8816300129366106, 0.7548512289780077)
                                                            ## ATD :      (4.654411764705882, 0.571524064171123, 0.19932659932659932, 0.47364864864864864))
      
    
## sensor_set = ['24', '10', '14', '30', '32', '13', '25', '7', '29', '22', '8', '26', '11', '28', '21', '12', '31', '9', '27', '15']  
                                   ## Top 20 node-locations (according to information theory)
                                              ## residual-data
                                                    ## Accuracy : (0.1032258064516129, 0.6761290322580645, 0.743225806451613,  0.29161290322580646)
                                                    ## ATD      : (3.982, 0.7694369973190348, 0.3977042538825118, 2.1042780748663104)
                                              ## cosine-data

 

# COMMAND ----------

# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
  
# X -> features, y -> label
X = pdf_simulation_results_cos[sensor_set].values
y = pdf_simulation_results_cos['LeakNode'].values
  
## dividing X, y into train and test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.01)  
  
X_train = X
X_test  = X
y_train = y
y_test  = y
  
# training a DescisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
accuracy_dtree = dtree_model.score(X_test, y_test)
cm_dtree = confusion_matrix(y_test, dtree_predictions)

# training a linear SVM classifier
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
accuracy_svm = svm_model_linear.score(X_test, y_test)
cm_svm = confusion_matrix(y_test, svm_predictions)

# training a KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
knn_predictions = knn.predict(X_test) 
accuracy_knn = knn.score(X_test, y_test)
cm_knn = confusion_matrix(y_test, knn_predictions)
  
# training a Naive Bayes classifier
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
accuracy_gnb = gnb.score(X_test, y_test)
cm_gnb = confusion_matrix(y_test, gnb_predictions)


# COMMAND ----------

accuracy_dtree, accuracy_svm, accuracy_knn, accuracy_gnb

# COMMAND ----------

#Average Topological Distance
ATD_dtree_numerator = 0
ATD_dtree_denumerator = 0
for i in range(0, len(X_columns)-1):
  for j in range(0, len(X_columns)-1):
    ATD_dtree_numerator   += cm_dtree[i][j] * topological_distance[X_columns[i]][X_columns[j]]
    ATD_dtree_denumerator += cm_dtree[i][j]
ATD_dtree = ATD_dtree_numerator / ATD_dtree_denumerator

ATD_svm_numerator = 0
ATD_svm_denumerator = 0
for i in range(0, len(X_columns)-1):
  for j in range(0, len(X_columns)-1):
    ATD_svm_numerator   += cm_svm[i][j] * topological_distance[X_columns[i]][X_columns[j]]
    ATD_svm_denumerator += cm_svm[i][j]
ATD_svm = ATD_svm_numerator / ATD_svm_denumerator

ATD_knn_numerator = 0
ATD_knn_denumerator = 0
for i in range(0, len(X_columns)-1):
  for j in range(0, len(X_columns)-1):
    ATD_knn_numerator   += cm_knn[i][j] * topological_distance[X_columns[i]][X_columns[j]]
    ATD_knn_denumerator += cm_knn[i][j]
ATD_knn = ATD_knn_numerator / ATD_knn_denumerator

ATD_gnb_numerator = 0
ATD_gnb_denumerator = 0
for i in range(0, len(X_columns)-1):
  for j in range(0, len(X_columns)-1):
    ATD_gnb_numerator   += cm_gnb[i][j] * topological_distance[X_columns[i]][X_columns[j]]
    ATD_gnb_denumerator += cm_gnb[i][j]
ATD_gnb = ATD_gnb_numerator / ATD_gnb_denumerator

# COMMAND ----------

ATD_dtree, ATD_svm, ATD_knn, ATD_gnb
