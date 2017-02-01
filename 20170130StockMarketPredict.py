# Simple Stock Market Predictor
# Built with Python 3.5

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvFile:
		csvFileReader = csv.reader(csvFile)
		start_time = time.time()
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	print("---- Completed reading file : %s in %s seconds ---" % (filename,time.time() - start_time))
	print(dates)
	return

def predict_prices(dates, prices, x):
	dates = np.reshape(dates,(len(dates),1))
	svr_lin = SVR(kernel = 'linear', C=1e3)
	svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
	svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)
	start_time = time.time()
	svr_lin.fit(dates,prices)
	print("--- Linear Fit Complete in %s seconds ---" % (time.time() - start_time))
	print("\n")
    
    # The Polynomial fitting did not complete within a reasonable time, therefore commenting it out.
	# svr_poly.fit(dates,prices)
	# print("Polynomial Fit Complete")
	start_time = time.time()
	svr_rbf.fit(dates,prices)
	print("--- RBF Fit Complete in %s seconds ---" % (time.time() - start_time))
	print("\n")


	rbf_prediction = svr_rbf.predict(x)[0], 
	linear_prediction = svr_lin.predict(x)[0]

	print("RBF Prediction is : ",rbf_prediction)
	print("\n")
	print("Linear Prediction is : ",linear_prediction)

	plt.scatter(dates,prices,color='black', label='Data')
	plt.plot(dates,svr_rbf.predict(dates), color = 'red', label = 'RBF model')
	plt.plot(dates,svr_lin.predict(dates), color = 'blue', label = 'Linear model')
	# plt.plot(dates,svr_poly.predict(dates), color = 'red', label = 'Polynomial model')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()
	return rbf_prediction, linear_prediction

get_data('ge.csv')

predicted_price = predict_prices(dates,prices,31)


