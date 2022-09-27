# Databricks notebook source
# MAGIC %md
# MAGIC ### 1 Setup of WDN & leak-scenario's

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

# COMMAND ----------

# Show/Vizualize the Water Network Model (Hanoi Optimized)
ax = wntr.graphics.plot_network(wn, node_attribute='base_demand', node_colorbar_label='Demand')

# COMMAND ----------

# Define simulation parameters 
start_time    = 2 * 3600 # 2 hours
leak_duration = 4 * 3600 # 4 hours
total_duration = start_time + leak_duration

leak_demand = [x * 3600 / 1000 for x in range(1,50+1)]  # m3/h

minimum_pressure  = 3.52 # 5 psi         ## to be checked = is this correct with our settings ??
required_pressure = 14.06 # 20 psi      ## to be checked = is this correct with our settings ??

min_pipe_diam = 0.1524 # 6 inch
max_pipe_diam = 0.2032 # 8 inch

Tau = 2.5 #pressure-drop threshold 


# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 Simulate SteadyStateconditions 

# COMMAND ----------

# Run Steady State simulation
sim = wntr.sim.WNTRSimulator(wn)
result_SteadyState = sim.run_sim()

# COMMAND ----------

result_SteadyState.node['demand'].round(decimals=2)

# COMMAND ----------

result_SteadyState.node['pressure'].round(decimals=1)

# COMMAND ----------

junct_of_interest = set(wn.node_name_list)
junct_of_interest.remove('1')          # there must be better ways to handle non-junctions, but ok for now

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 Simulate LeakScenario's

# COMMAND ----------

# Create dictionary to save results
analysis_results = {}

# Simulate for each lead_demand level
for leak_demand_m3h in leak_demand:
  leak_demand_m3s = leak_demand_m3h / 3600 # m3/s
  
  # Create dictionary to save results per leak demand level
  analysis_results_per_leak_demand_level = {}
  
  # Simulate fire flow demand for each hydrant location
  for junct in junct_of_interest:
      wn = wntr.network.WaterNetworkModel(inp_file_name)
      wn.options.hydraulic.demand_model = 'PDD'    
      wn.options.time.duration = total_duration
      wn.options.hydraulic.minimum_pressure = minimum_pressure
      wn.options.hydraulic.required_pressure = required_pressure
  
      # Create leak flow pattern
      leak_flow_pattern = wntr.network.elements.Pattern.binary_pattern(
          'leak_flow',
          start_time=start_time,
          end_time=total_duration,
          step_size=wn.options.time.pattern_timestep,
          duration=wn.options.time.duration
          )
      wn.add_pattern('leak_flow', leak_flow_pattern)
  
      # Apply fire flow pattern to hydrant location
      leak_junct = wn.get_node(junct)
      leak_junct.demand_timeseries_list.append((leak_demand_m3s, leak_flow_pattern, 'Leak flow'))
  
      try:
          # Simulate hydraulics
          sim = wntr.sim.WNTRSimulator(wn) 
          sim_results = sim.run_sim()
   
          HeadLoss = (result_SteadyState.node['pressure'].round(decimals=1).iloc[0] - sim_results.node['pressure'].round(decimals=1).iloc[3]).to_dict()
             
      except Exception as e:
          # Identify failed simulations and the reason
          HeadLoss = None
          print(junct, ' Failed:', e)
  
      finally:
          # Save simulation results
          analysis_results_per_leak_demand_level[junct] = HeadLoss
          
  analysis_results[leak_demand_m3h] = analysis_results_per_leak_demand_level

# COMMAND ----------

pdf1 = pd.DataFrame.from_dict(analysis_results, orient='index').stack().to_frame()
pdf1_analysis_results = pd.DataFrame(pdf1[0].values.tolist(), index=pdf1.index).rename_axis(['LeakDemand', 'LeakNode'])
pdf_simulation_results = pd.DataFrame(pdf1_analysis_results.stack().reset_index().rename(columns={"level_2": "SensorLocation", 0: "PressureDrop"}))
sdf_simulation_results=spark.createDataFrame(pdf_simulation_results)

# COMMAND ----------

# Store results of pressure-drop simulation for later use
dbutils.fs.rm("dbfs:/mnt/dlwadlsgen2/waterlink/99QuantumSensorPlacementDEL20/simulation_results", True)
sdf_simulation_results.write.format("delta")\
                            .mode("overwrite")\
                            .save("dbfs:/mnt/dlwadlsgen2/waterlink/99QuantumSensorPlacementDEL20/simulation_results")

# COMMAND ----------

pdf1_analysis_results
