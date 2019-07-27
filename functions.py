import os
import math as m
import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
# ---------------------------------------------------------
#import statsmodels.tsa.api as smt
# ---------------------------------------------------------


#data to read
dir_path = os.path.dirname(os.path.abspath(__file__))
#dataset_path = dir_path + '/data/'
dataset_path = dir_path + '\\data\\'
file_paths = [dataset_path + 'aet.csv',

                dataset_path + 'cat.csv',
                dataset_path + 'cop.csv',
                dataset_path + 'dis.csv',
                dataset_path + 'gis.csv',
                dataset_path + 'mat.csv',
                dataset_path + 'ko.csv',
                dataset_path + 'mas.csv',
                dataset_path + 'noc.csv',
                dataset_path + 'oxy.csv',
                dataset_path + 'pki.csv',
                dataset_path + 'rok.csv',
                dataset_path + 'uis.csv',
                dataset_path + 'wfc.csv',
                dataset_path + 'wmt.csv',
                dataset_path + 'xom.csv',
                dataset_path + 'xrx.csv',]

'''






[dataset_path + 'ABC_data.csv',

                dataset_path + 'HES_data.csv',

                dataset_path + 'NVDA_data.csv',
                dataset_path + 'FCX_data.csv',
                dataset_path + 'GOOGL_data.csv',
                dataset_path + 'BIIB_data.csv',
                dataset_path + 'UAA_data.csv',
                dataset_path + 'VRSK_data.csv',
                dataset_path + 'SJM_data.csv',
                dataset_path + 'TMK_data.csv', 
        dataset_path + 'KEY_data.csv',
        dataset_path + 'GPS_data.csv',
        dataset_path + 'HRB_data.csv',
        dataset_path + 'IRM_data.csv',
        dataset_path + 'KORS_data.csv',
        dataset_path + 'MCK_data.csv',
        dataset_path + 'MMC_data.csv',
        dataset_path + 'MTD_data.csv',
        dataset_path + 'PDCO_data.csv',
        dataset_path + 'STT_data.csv'] 
            

[dataset_path + 'banknifty.csv']

'''


                

input_data = "AR"
Raw_data=[]
closing_prices = []
AR_predictions = []
AR_test_predictions =[]
AR_residuals = []
AR_predictions_test = []
closing_prices_test = []
AR_residuals_test = []
AR_mdl = []
ARIMA_mdl=[]
ARIMA_predictions=[]
ARIMA_residuals=[]
ARIMA_test_predictions=[]
ARIMA_test_residuals=[]
predictions=[]
num_of_rows_to_read = 0


i = 0
print(file_paths[i])

max_lag = 10
ma_params = 0
returns=True
log=True

s_ups=[]


def get_returns(x):
    x_returns=[]
    ups=0
    x_returns.append(0)
    for i in range(len(x)-1):
        x_returns.append(((x[i+1]-x[i])/x[i]))
        if(x[i+1]>x[i]):
            ups=ups+1
    
    ups= (ups*1.0/len(x_returns))
    print("Ups: ",ups)
    return np.asarray(x_returns),ups

def get_returns_log(x):
    x_returns=[]
    ups=0
    x_returns.append(0)
    for i in range(len(x)-1):
        x_returns.append(m.log(x[i+1]/x[i]))
        if (x[i+1]>x[i]):
            ups=ups+1
    ups= (ups*1.0/len(x_returns))
    print("Ups: ",ups)
    return np.asarray(x_returns),ups


def raw_lag(lags,cl_prices):

    L = []
    i=0
    for j in range(lags, len(cl_prices[i])-1):
        ar = []
        for i in range(len(cl_prices)):
            for k in range(j-lags,j):
                ar.append(cl_prices[i][k])
        L.append(ar)

    return L

def log_reg(lags,cl_prices):
    log_models=[]
    x_data=[]
    for i in range(len(cl_prices)):
        input_data=[]
        for j in range(lags,len(cl_prices[i])):
            input_data.append(cl_prices[i][j-lags:j])
        x_data.append(input_data)
        
    return x_data
    

def s_normalize(ar):
    m=sum(ar)/len(ar)
    ar=(ar-m)/np.std(ar)
    return ar

def s_normalize2(ar):
    ar=ar/np.std(ar)
    return ar

def normalize_min_max(ar):
    
    ar=(ar)/((max(ar)-min(ar))/2)
    return ar



def read_data(input_d, *date):
    #reading the csv files for training set filling
    
    print("Lags: ",max_lag)
    print("samples: ",num_of_rows_to_read)
    print("Stocks num: ",len(file_paths))
    input_data=input_d
    start_date='1985-01-01'
    end_date='2017-01-01'

        
    if len(date)>0:
        start_date=date[0]
        end_date=date[1]
    


    for i in range(len(file_paths)):
        #df = pd.read_csv(file_paths[i])
        df = pd.read_csv(file_paths[i],index_col = 0)
        cl_price = df['Close']
        cl_price = cl_price[start_date:end_date]
        # Create x, where x the 'scores' column's values as floats


        x = cl_price.values
        if(returns):
            if(log):
                x,ups = get_returns_log(x)
                x = s_normalize2(x)
            else:
                x,ups = get_returns(x)
                x=s_normalize2(x)
                
            if(ups>=0.5):
                s_ups.append(ups)
            else:
                s_ups.append(1-ups)
            
         # Run the normalizer on the dataframe
        
        closing_price = pd.DataFrame(x)

        #Test set predictions
        size = int(len(closing_price) * 0.80)
        train, test = closing_price[0:size][0].values, closing_price[size:len(closing_price)][0].values
        
        closing_prices.append(train)
        closing_prices_test.append(test)
    
    
    if(input_data=="ARIMA"):
        input_train=ARIMA_predictions
        input_test=ARIMA_test_predictions

    elif(input_data=="AR"):
        input_train=AR_predictions
        input_test=AR_test_predictions
    
    elif(input_data=="AR_res"):
        input_train=AR_residuals
        input_test=AR_residuals_test

    elif (input_data =="LOG_REG"):
        input_train=log_reg(max_lag,closing_prices)
        input_test=log_reg(max_lag,closing_prices_test)

    elif(input_data=="RAW"):
        input_train=raw_lag(max_lag,closing_prices)
        print("Raw input shape: ",np.shape(input_train))
        input_test=raw_lag(max_lag,closing_prices_test)
        print("Raw input shape test: ",np.shape(input_test))
    
    print("Ups Average: ", sum(s_ups)/len(s_ups))   
    print("ok")
    #plot
    #plt.plot(input_train[0],label='input_train'+str(0))
    #plt.plot(closing_prices[0],label='closing_'+str(0))
    #plt.plot(input_test[0],label='input_test_'+str(0))
    #plt.plot(closing_prices_test[0],label='closing_test_'+str(0))
    
    #plt.legend()
    #plt.show()


    #Ziping lines->columns & clomns->lines 
    #train & test
    #So they match the form on DNN
    
    zclosing_prices = list(zip(*closing_prices))
    zclosing_prices[0:max_lag]=[]
    
    zclosing_prices_test = list(zip(*closing_prices_test))
    zclosing_prices_test[0:max_lag]=[]

    if(input_data != "RAW"):
        z_input_train = list(zip(*input_train))
        z_input_test = list(zip(*input_test))


    else:
        z_input_train=input_train
        #z_input_train.pop()
        z_input_test=input_test
        #z_input_test.pop()
    return(z_input_train, zclosing_prices,z_input_test, zclosing_prices_test,len(file_paths),max_lag,s_ups)

