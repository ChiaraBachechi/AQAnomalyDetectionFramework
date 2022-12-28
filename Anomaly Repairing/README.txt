###################################### README #####################################

To execute the anomaly repairing on a dataset you need firstly to aggregate values
every ten minutes and create a dataset with null values where anomalies have been
detected or observations are missing.
Then run the python script:

	python AnomalyRepairing.py -f <input_file_name> -o <output_file_name>

The script takes as parameters:

-f <input_file_name> 	 the name of the input file:
			(a .csv file) with all the aggregated raw observations
			both anomalies and valid values. The values that are 
	 		consiered anomalous should be set to the 'NaN' value.


-o <output_file_name>   the name of the output file:
			(a .csv file) with only the repaired observations.

Example:
	python AnomalyRepairing.py -f to_be_repaired.csv -o repaired.csv

#################################################################################

The structure of the input file is:

id_sensor_low_cost_status: 			
		this value identify the sensor itself or the sensor in a precise position. 
		Observations with the same status are considered as a unique time series. 
		If the status change the time series is different.

phenomenon_time_sensor_raw_observation_10min: 
		this value is the timestamp of the observation. 
		In our use case, we have an observation every 10 minutes 
		but the code will work with any other interval.

A column for each observed value:
In our use case:
	no_we
	no2_we
	no_aux
	no2_aux
	ox_aux
	ox_we
	co_we
	co_aux
	temperature
	humidity	
Example:
	file to_be_repaired.csv in this folder.