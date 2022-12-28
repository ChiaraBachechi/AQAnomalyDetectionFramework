#!/usr/bin/env python
# coding: utf-8


import argparse
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from statsmodels.tsa.api import VAR
from numpy.matlib import repmat
import pandas.io.sql as sqlio
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

def addOptions():
	parser = argparse.ArgumentParser(description='Repair missing observations.')
	parser.add_argument('--file-name', '-f', dest='file_name', \
							help="""The CSV file where the measurements of low-cost sensors are collected.""")
	parser.add_argument('--out-file-name', '-o', dest='out_file_name', \
							help="""The CSV file where the repaired data will be inserted.""")
	return parser	

def invert_transformation(df_train, df_forecast, second_diff=False):
	"""Revert back the differencing to get the forecast to original scale."""
	df_fc = df_forecast.copy()
	columns = df_train.columns
	for col in columns:		
		# Roll back 2nd Diff
		if second_diff:
			df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
		# Roll back 1st Diff
		df_fc[str(col)] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
	return df_fc


def forecast_accuracy(forecast, actual):
	mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
	me = np.mean(forecast - actual)			 # ME
	mae = np.mean(np.abs(forecast - actual))	# MAE
	mpe = np.mean((forecast - actual)/actual)   # MPE
	rmse = np.mean((forecast - actual)**2)**.5  # RMSE
	corr = np.corrcoef(forecast, actual)[0,1]   # corr
	return({'mape':mape, 'me':me, 'mae': mae, 
			'mpe': mpe, 'rmse':rmse, 'corr':corr})



def Repairing(df):
	sumMAPElast = 0
	countLast = 0
	not_repaired = []
	df_result = pd.DataFrame()
	df_test_tot = pd.DataFrame()
	starting_time = datetime.now()
	for status in df['id_sensor_low_cost_status'].unique().tolist():
		print('status:')
		print(status)
		df = df[df.id_sensor_low_cost_status == status]
		#df = df.set_index('phenomenon_time_sensor_raw_observation_10min')
		index_buchi = df[df.isnull().any(axis=1) == True].index.values
		for index_buco in index_buchi:
			df_test = df[df.index.values == index_buco]
			df_train = df[df.index.values < index_buco].dropna()
			df_train.drop(['id_sensor_low_cost_status'],axis = 1,inplace = True)
			if (df_train.shape[0] < 10):
				print('The dimension of the dataset is too small. No repair will be performed')
				continue
			df_differenced = df_train.drop(['humidity','temperature'],axis = 1)
			model = VAR(df_differenced)
			prev = model.fit(1)
			#tuning of p parameter
			p = 1
			for i in [1, 2, 3,4,5,6,7,8]:
				result = model.fit(i)
				p = i
				print(str(i)+ ':')
				print(result.aic)
				if (prev.aic - result.aic) < 0:
					p = i-1
					break
			print('value of p:'+ str(p))
			differenciate = False
			double = False
			try:
				model_fitted = model.fit(maxlags=5, ic='aic')
			except(Exception) as error:
				df_differenced = df_differenced.diff().dropna()
				model = VAR(df_differenced)
				differenciate = True
				try:
					model_fitted = model.fit(maxlags=5, ic='aic')
				except(Exception):
					df_differenced = df_differenced.diff().dropna()
					double = True
					model = VAR(df_differenced)
					try:
						model_fitted = model.fit(maxlags=5, ic='aic')
					except(Exception):
						item = {}
						item['id_sensor_low_cost_status'] = status
						item['datetime'] = index_buco
						#print(item)
						continue
			out = durbin_watson(model_fitted.resid)
			# for col, val in zip(df_differenced.columns, out):
				# print(col, ':', round(val, 2))
			lag_order = model_fitted.k_ar
			print(lag_order)  #> 4
			# Input data for forecasting
			forecast_input = df_differenced.values[-lag_order:]
			fc = model_fitted.forecast(y=forecast_input, steps=1)
			if double:
				strcol = '_2d'
			elif differenciate:
				strcol = '_1d'
			else:
				strcol = ''
			df_forecast = pd.DataFrame(fc, index=df_test.index, columns=df_differenced.columns + strcol)
			df_sub = pd.DataFrame(fc, index=df_test.index, columns=df_differenced.columns + strcol)
			if differenciate:
				df_forecast = invert_transformation(df_train, df_forecast, second_diff = double)
			accuracy = []
			accuracy_last = []
			for x in df_forecast.columns:
				if x.find('1d') >= 0 or x.find('2d') >= 0:
					df_forecast.drop([x],axis=1,inplace=True)
				else:
					if df_test[x].isnull().any() == False:
						accuracy.append(forecast_accuracy(df_forecast[x].values, df_test[x])['mape'])
						print(df_train[x].iloc[[-1]])
						accuracy_last.append(forecast_accuracy(df_train[x].iloc[-1], df_test[x])['mape'])
						df_sub[x] = df_test[x]
					else:
						df_sub[x] = df_forecast[x]
			df_forecast['id_sensor_low_cost_status'] = status
			meanMape = float('NaN')
			if len(accuracy) == 0:
				n_null = df[(df.phenomenon_time_sensor_raw_observation_10min < index_buco) & 
					   (df.phenomenon_time_sensor_raw_observation_10min > (index_buco - np.timedelta64(1,'h')))  ].isna().any(axis=1).sum()
				if n_null >= 5:
					print('too much consecutive anomalies: no repairing is performed.')
					continue
			else:
				meanMape = sum(accuracy) / len(accuracy)
				if sum(accuracy) / len(accuracy) > 0.5:
					if sum(accuracy_last) / len (accuracy_last) > 0.5:
						continue
					else:
						for x in df_sub.columns:
							df_sub[x] = df_train.iloc[[-1]][x].values[0]
						sumMAPElast = sumMAPElast + sum(accuracy_last) / len (accuracy_last)
						countLast = countLast + 1
						for x in df_sub.columns:
							df_forecast.at[index_buco,x] = df_sub.iloc[0][x]
			# print('test')
			# print(df_test.to_string())
			# print('value to substitue')
			# print(df_sub.to_string())
			# print('forecasted value')
			# print(df_forecast.to_string())
			df_sub['temperature'] = df_test['temperature']
			df_sub['humidity'] = df_test['humidity']
			df_sub['id_sensor_low_cost_status'] = status
			for x in df_sub.columns:
				df.at[index_buco,x] = df_sub.iloc[0][x]
			df_forecast['differenciated'] = differenciate
			df_forecast['doubled'] = double
			df_forecast['MAPE'] = meanMape
			df_result = pd.concat([df_result, df_forecast])
			df_result = pd.concat([df_result, df_forecast])
			df_test_tot = pd.concat([df_test_tot, df_test])
	ending_time = datetime.now()
	duration = ending_time - starting_time						# For build-in functions
	duration_in_s = duration.total_seconds()
	print('Duration of the repairing process in seconds: ' + str(duration_in_s))
	return df_result


def main(args=None):
	argParser = addOptions()
	options = argParser.parse_args(args = args)
	inputFileName = options.file_name
	outputFileName = options.out_file_name
	df_input = pd.read_csv(inputFileName, sep =',')
	df_input.set_index('phenomenon_time_sensor_raw_observation_10min', inplace = True)
	print(df_input.columns)
	print(df_input.shape)
	df_repaired = Repairing(df_input.copy())
	df_repaired.to_csv(outputFileName)
	return 0


main()





