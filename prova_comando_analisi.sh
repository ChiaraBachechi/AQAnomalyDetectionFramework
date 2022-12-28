#!/bin/sh
python3 main.py --action AnomalyDetectionHistoricalWriteCSV --id_algorithm 2 --begin_time "2019-08-01 00:00:00" --end_time "2019-08-10 00:00:00" --plot True --id_sensor 4007;
python3 main.py --action AnomalyDetectionHistoricalWriteCSV --id_algorithm 2 --begin_time "2019-08-01 00:00:00" --end_time "2019-09-10 00:00:00" --plot True --id_sensor 4008;
