import os
import math as m
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

 




#path variables
dir_path = os.path.dirname(os.path.abspath(__file__))


savefig_dir = dir_path+'\\random_forest_figures\\'



# import functions file
functions = dir_path + '\\functions.py'
from functions import read_data



#input data form: AR predictions / AR residuals / ARIMA predictions / RAW laggs 
input_d="LOG_REG"
returns= True
start_date='2010-01-01'
end_date='2017-01-01'
# read csv
x_data, y_data,x_test, y_test,stocks_num,lags,ups = read_data(input_d,start_date,end_date)

print("Shapes: ", np.shape(x_data),np.shape(y_data),np.shape(x_test),np.shape(y_test))


x_data=np.asarray(x_data)
y_data=np.asarray(y_data)
x_test=np.asarray(x_test)
y_test=np.asarray(y_test)

y= y_data >= 0
y_test = y_test >=0

predictions=[]
models=[]
accuracy=[]
for i in range(len(x_data[0])):
	model = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
	model.fit(x_data[:, i], y[:,i])
	
	models.append(model)
	prediction = model.predict(x_test[:,i])
	predictions.append(prediction)
	accuracy.append(accuracy_score(y_test[:,i], predictions[i]))
	print("Accuracy score:", accuracy[i])


print("---------------------------------------------------")
avg_accuracy=sum(accuracy)/len(accuracy)

print("Avg Accuracy score:", avg_accuracy)
print("---------------------------------------------------")
accuracy.append(avg_accuracy)
ups.append(sum(ups)/len(ups))


Accuracy= str(round(accuracy[-1], 3))
Up=str(round(ups[-1],3))


x = np.arange(1,len(accuracy)+1)
plt.bar(x,accuracy,0.2,label='Random Forest ('+start_date+'..'+end_date+')')
plt.bar(x+0.2, ups,0.2, label='ups/downs percentage')
plt.xticks(x, ('1', '2', '3', '4','5', '6', '7', '8','9', '10', '11', '12','13', '14', '15', '16','17','avg'))
plt.legend(loc=4)

#Save  figure
num_fig_files = len([f for f in os.listdir(savefig_dir)if os.path.isfile(os.path.join(savefig_dir, f))])+1
#results_file_name = savefig_dir+"/"+str(num_fig_files)+"_result_RF_"+Accuracy+"_ups_"+Up+"_a.png"
results_file_name = savefig_dir+"/RF_"+Accuracy+"_ups_"+Up+"_a.png"
plt.savefig(results_file_name)

plt.show()




difference=[]
for i in range(len(accuracy)):
	difference.append(accuracy[i]-ups[i])

plt.bar(x,difference,0.4,label='Random Forest, ups/downs difference ('+start_date+'..'+end_date+')')
plt.xticks(x, ('1', '2', '3', '4','5', '6', '7', '8','9', '10', '11', '12','13', '14', '15', '16','17','avg'))
plt.legend()
#Save  figure

#results_file_name = savefig_dir+"/"+str(num_fig_files)+"_result_RF_"+Accuracy+"_ups_"+Up+"_b.png"
results_file_name = savefig_dir+"/RF_"+Accuracy+"_ups_"+Up+"_b.png"

plt.savefig(results_file_name)

plt.show()



