#cleaning data

import pandas as pd
import numpy as np
from scipy import optimize
from utility_functions import CheckUnique, CreateBinary, encode_onehot, Check_same_values, PCA_compress, CheckNegativeVal, CreateChangeFeature
from sklearn.preprocessing import RobustScaler, MinMaxScaler

#from collections import defaultdict
#from datetime import date, datetime
#import warnings
#from itertools import combinations

folder = '/home/nuc/Desktop/PythonProgs/FlowRecorder/DuncanData/Full_Data/'

#split the dataset to morning and night here!

norm_data = pd.read_csv(folder+'normal_data.csv')
test_data = pd.read_csv(folder+'anomaly_data_latency.csv')

norm_data2 = pd.read_csv(folder+'normal_data2.csv')
test_data2 = pd.read_csv(folder+'anomaly_data_battery.csv')
#1) During day time, battsV<13V is anomaly. During night <12.5V is anomaly. Night is statenum=3
#2) battV-BattsenseV>0.1 is anomaly
#3) take out error data 159.87
#4) column C >14.4 is anom?

norm_data = CreateChangeFeature(norm_data,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
norm_data2 = CreateChangeFeature(norm_data2,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
test_data = CreateChangeFeature(test_data,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
test_data2 = CreateChangeFeature(test_data2,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])

#normal operations data taken. those that have a few discrepancies are removed for training.
df = pd.concat([norm_data2[:1300],norm_data2[1600:3200],norm_data2[3500:3800],norm_data2[4400:5100],norm_data2[5600:5900],norm_data2[7100:8400],norm_data2[10200:10800],norm_data2[11800:12500]]).reset_index(drop=True)

norm_data_full = norm_data.append(df,ignore_index=True).reset_index(drop=True)
test_data_full = test_data.append(test_data2,ignore_index=True).reset_index(drop=True)
del norm_data, norm_data2, test_data, test_data2, df

norm_data = norm_data_full.drop(norm_data_full[norm_data_full['battsI']>120].index,axis=0).reset_index(drop=True)
test_data = test_data_full.drop(test_data_full[test_data_full['battsI']>120].index,axis=0).reset_index(drop=True)

#still checking on 'signal', 'TXLatency'
#'TXRate', 'RXRate' have some problems in the test file. Removed for now.
#List of numerical attributes in the data. Names need to match the dataset
numerical = ['battsV', 'battsSensedV', 'battsI', 'arrayV', 'arrayI',
       'v_target', 'hsTemp', 'rtsTemp', 'outPower', 'inPower', 'sweep_pmax',
       'sweep_vmp', 'sweep_voc', 'minVb_daily', 'maxVb_daily', 'ahc_daily',
       'whc_daily', 'minTb_daily', 'maxTb_daily', 'weather_temp', 'weather_wind',
        'latency','TXCapacity', 'RXCapacity','signal', 'SNR',
        'NoiseFloor','TXLatency']

#'flags_daily' is removed for now
#list of nominal variables in the data
nominal = ['statenum','weather_long']

#list of indicator variables which have a fixed value for data
fixed_cols = ['dipswitches','CCQ','freq'] #fixed value. anomaly if changes

#convert datatype in nominal variables to string if they are not already
for col in nominal:
    norm_data[col] = norm_data[col].astype(str)
    test_data[col] = test_data[col].astype(str)

#remove rows with latency above 10 and TXLatency above 10. Latency values above 10 are anomalies that we want to detect.
norm_data = norm_data.drop(norm_data[norm_data['latency']>10].index).reset_index(drop=True)
norm_data = norm_data.drop(norm_data[norm_data['TXLatency']>10].index).reset_index(drop=True)

#label anomaly data
test_data['label'] = 0
test_data.loc[test_data[test_data['latency']>10].index,'label']=1


#create binary feature for fixed values
CheckUnique(norm_data,fixed_cols)

vals = {}
for col in fixed_cols:
    vals[col] = norm_data[col].unique()[0]

norm_data = CreateBinary(norm_data,fixed_cols,vals)
test_data = CreateBinary(test_data,fixed_cols,vals)

#check normal data in test dataset
des_norm_train = norm_data.describe()
des_norm_test = test_data[test_data['label']==0].describe()
des_diff = des_norm_train - des_norm_test #numbers should be small for all columns
#print(des_diff)

#one hot encode to transform nominal features to binary
norm_data_enc = encode_onehot(norm_data, nominal)
test_data_enc = encode_onehot(test_data, nominal)

Check_same_values(norm_data,test_data,norm_data_enc,test_data_enc,colms=nominal)
#check if need to merge some colums, eg, freq, CCQ, dipswitches

#list of binary variables after one hot encoding
binary = list(norm_data_enc.columns[list(norm_data_enc.columns).index('dipswitchesf'):])

CheckUnique(norm_data_enc,binary) #should not produce any output
CheckUnique(test_data_enc,binary)

#in this dataset, CCQ=333, dipswitches=101111 and freq=5645 has only one value in both training and testing set
#so pointless but we are still including it here for completeness

########################################################################################################################
#create new feature

#delta_features=['ahc_daily']  #currently not in use
#norm_data_enc = CreateChangeFeature(norm_data_enc,delta_features)
#numerical.extend(list(norm_data_enc.columns)[-len(delta_features):])

##########################################################################################################################3
#preprocess


def PreProcess1(norm_data_enc, test_data_enc):
    np.random.seed(10)
    #use pca to convert binary columns to numerical.
    scaled_pca_bin_normal_tr, scaled_pca_bin_ts = PCA_compress(norm_data_enc, test_data_enc, cols=binary)

    #for numerical columns
    CheckNegativeVal(numerical,norm_data_enc) #most network data should not have negative data
    #signal and NoiseFloor are negative values by default. We can map them to positive numbers
    #the ratio between each value remains constant. The positive value is important for our choice of normalisation
    norm_data_enc['signal'] = -1*norm_data_enc['signal']
    norm_data_enc['NoiseFloor'] = -1*norm_data_enc['NoiseFloor']
    test_data_enc['signal'] = -1*test_data_enc['signal']
    test_data_enc['NoiseFloor'] = -1*test_data_enc['NoiseFloor']

    CheckUnique(norm_data_enc,numerical)

    #normalising the data. The following method only works for positive values.
    #normalize numerical cols to range (0,1) by mapping mean of the feature value to 0.5
    def f(x,val=0):
        return (1 - np.exp(-x*val))/(1 + np.exp(-x*val)) - 0.5

    def get_k_val(means,cols = numerical):
        k_val = pd.DataFrame(columns=['k_val'],index=cols)

        for col in cols:
            val = means[col]
            try:
                root = optimize.brentq(f, 0, 100000,val)
                #print('{} : {}'.format(col,root))
                k_val['k_val'].loc[col]=root
            except:
                print(col)
        return k_val
    k_val = get_k_val(norm_data_enc[numerical].mean(),cols = numerical) #transmit k_val back to site for normalizing

    #normalize numerical column in each site
    Normalized_training_df= pd.DataFrame(columns=numerical)
    for col in numerical:
        Normalized_training_df[col] = f(norm_data_enc[col], k_val['k_val'].loc[col]) + 0.5
    Normalized_training_df = pd.concat([Normalized_training_df,scaled_pca_bin_normal_tr],axis=1)


    Normalized_testing_df= pd.DataFrame(columns=numerical)
    for col in numerical:
        Normalized_testing_df[col] = f(test_data_enc[col],k_val['k_val'].loc[col])+0.5

    Normalized_testing_df = pd.concat([Normalized_testing_df,scaled_pca_bin_ts],axis=1)
    return Normalized_training_df,Normalized_testing_df




def Preprocess(norm_data_enc,test_data_enc,method='MinMax Scaling'):

    np.random.seed(10)
    #use pca to convert binary columns to numerical.
    scaled_pca_bin_normal_tr, scaled_pca_bin_ts = PCA_compress(norm_data_enc, test_data_enc, cols=binary)

    if method=='Robust Scaling':
        scaler = RobustScaler().fit(norm_data_enc[numerical])
    elif method=='MinMax Scaling':
        scaler = MinMaxScaler().fit(norm_data_enc[numerical])

    Normalized_training_df = pd.DataFrame(scaler.transform(norm_data_enc[numerical]),columns=numerical)
    Normalized_testing_df = pd.DataFrame(scaler.transform(test_data_enc[numerical]),columns=numerical)
    Normalized_training_df = pd.concat([Normalized_training_df, scaled_pca_bin_normal_tr], axis=1)
    Normalized_testing_df = pd.concat([Normalized_testing_df, scaled_pca_bin_ts], axis=1)

    return Normalized_training_df,Normalized_testing_df

labels = test_data_enc['label']
#Normalized_training_df,Normalized_testing_df=PreProcess1(norm_data_enc, test_data_enc)
Normalized_training_df,Normalized_testing_df = Preprocess(norm_data_enc,test_data_enc,method='MinMax Scaling')


cols = Normalized_training_df.columns
feature_sel_df,_ = Preprocess(test_data_enc,norm_data_enc,method='MinMax Scaling')
from sklearn.feature_selection import chi2
chi,pval = chi2(feature_sel_df[cols],labels)
df_rank = pd.DataFrame()
df_rank['features'] = cols
df_rank['chi'] = chi
df_rank = df_rank.sort_values('chi').reset_index(drop=True)
