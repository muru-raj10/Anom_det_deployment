#similar code to deployment but with graphs

import pandas as pd
import numpy as np
from scipy import optimize
from utility_functions import CheckUnique, CreateBinary, encode_onehot, Check_same_values, PCA_compress, CheckNegativeVal, CreateChangeFeature
import datetime
from sklearn.preprocessing import RobustScaler, MinMaxScaler

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utility_functions import evaluate_rate, classify_pts
import time
import random
import matplotlib.pyplot as plt
#from sklearn.svm import OneClassSVM
#from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score




#split the dataset to morning and night here!

norm_data = pd.read_csv(folder+'normal_data.csv')
test_data = pd.read_csv(folder+'anomaly_data_latency.csv')

norm_data2 = pd.read_csv(folder+'normal_data2.csv')
test_data2 = pd.read_csv(folder+'anomaly_data_battery.csv')

norm_data = CreateChangeFeature(norm_data,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
norm_data2 = CreateChangeFeature(norm_data2,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
test_data = CreateChangeFeature(test_data,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
test_data2 = CreateChangeFeature(test_data2,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])

#normal operations data taken. those that have a few discrepancies are removed for training.
df = norm_data2
df2 = test_data2

norm_data_full = norm_data.append(df,ignore_index=True).reset_index(drop=True)
test_data_full = test_data.append(df2,ignore_index=True).reset_index(drop=True)


norm_data_full = norm_data_full.drop(norm_data_full[norm_data_full['battsI']>120].index,axis=0).reset_index(drop=True)
test_data_full = test_data_full.drop(test_data_full[test_data_full['battsI']>120].index,axis=0).reset_index(drop=True)

def ConvertToDateTime(df):
    d_dtime_list = []
    d_time_list = []
    for i in range(len(df)):
        d_dtime_list.append(datetime.datetime.strptime(df['datetime'][i],"%d/%m/%Y %H:%M:%S"))
        d_time_list.append(datetime.datetime.strptime(df['datetime'][i], "%d/%m/%Y %H:%M:%S").time())
    df['DateTime'] = d_dtime_list
    df['Time'] = d_time_list
    return df

norm_data_full = ConvertToDateTime(norm_data_full)
test_data_full = ConvertToDateTime(test_data_full)

numerical = ['battsI', 'arrayV', 'arrayI',
       'v_target', 'hsTemp', 'rtsTemp', 'outPower', 'inPower', 'sweep_pmax',
       'sweep_vmp', 'sweep_voc', 'minVb_daily', 'maxVb_daily', 'ahc_daily',
       'whc_daily', 'minTb_daily', 'maxTb_daily', 'weather_temp', 'weather_wind',
        'TXCapacity', 'RXCapacity','signal', 'SNR',
        'NoiseFloor', 'ch_arrayV',
       'ch_arrayI', 'ch_hsTemp', 'ch_rtsTemp', 'ch_outPower', 'ch_inPower','TXLatency','latency'] #'battsV', 'battsSensedV','ch_battsV', 'ch_battsSensedV'

nominal = ['statenum','weather_long']
fixed_cols = ['dipswitches','CCQ'] 

#convert datatype in nominal variables to string if they are not already
for col in nominal:
    norm_data_full[col] = norm_data_full[col].astype(str)
    test_data_full[col] = test_data_full[col].astype(str)


#label anomaly data
test_data_full['label'] = 0
test_data_full.loc[test_data_full[test_data_full['latency']>10].index,'label']=1
labels_latency = test_data_full['label']

CheckUnique(norm_data_full,fixed_cols)

vals = {}
for col in fixed_cols:
    vals[col] = norm_data_full[col].unique()[0]

norm_data = CreateBinary(norm_data_full,fixed_cols,vals)
test_data = CreateBinary(test_data_full,fixed_cols,vals)

#######################################################################################################################3
#split data into time periods
def Split_by_Time(df,starth,endh):
    varble = []
    for i in range(len(df)):
        if df['Time'][i]>=datetime.time(starth,00):
            if df['Time'][i]<=datetime.time(endh,00):
                varble.append(1)
            else:
                varble.append(0)
        else:
            varble.append(0)
    col_name = str(starth)+'_'+str(endh)
    df[col_name] = varble
    return df

norm_data = Split_by_Time(norm_data,6,9)
norm_data = Split_by_Time(norm_data,9,11)
norm_data = Split_by_Time(norm_data,11,13)
norm_data = Split_by_Time(norm_data,13,15)
norm_data = Split_by_Time(norm_data,15,18)
norm_data = Split_by_Time(norm_data,18,21)


test_data = Split_by_Time(test_data,6,9)
test_data = Split_by_Time(test_data,9,11)
test_data = Split_by_Time(test_data,11,13)
test_data = Split_by_Time(test_data,13,15)
test_data = Split_by_Time(test_data,15,18)
test_data = Split_by_Time(test_data,18,21)

##############################################


#one hot encode to transform nominal features to binary
norm_data_enc = encode_onehot(norm_data, nominal)
test_data_enc = encode_onehot(test_data, nominal)

Check_same_values(norm_data,test_data,norm_data_enc,test_data_enc,colms=nominal)

#list of binary variables after one hot encoding
binary = list(norm_data_enc.columns[list(norm_data_enc.columns).index('Time')+1:])

CheckUnique(norm_data_enc,binary) #should not produce any output
CheckUnique(test_data_enc,binary)



##########################################################################################################################3
#preprocess

def Preprocess(norm_data_enc,test_data_enc,method='Robust Scaling'):

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
###########################################################################################################################

def TrainAE(Normalized_training_df,Normalized_testing_df,scaled='Robust Scaling'):

    random.seed(30)
    np.random.seed(30)
    print(len(Normalized_training_df))

    #convert data into tensor
    dataset = torch.tensor(Normalized_training_df.values)
    testset = torch.tensor(Normalized_testing_df.values)

    batchsize = int(0.05*len(Normalized_training_df))  #(1/20 th of the training dataset)
    #batchsize = 500

    trainloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1) #load with respect to batch size
    trainloader_all = DataLoader(dataset, len(Normalized_training_df), shuffle=True, num_workers=1) #load all data for score calculation
    testloader = DataLoader(testset, len(Normalized_testing_df), shuffle=False, num_workers=1)

    dim =np.shape(Normalized_training_df)[1]
    print(dim)
    latent_dim = int(round(np.sqrt(dim)+1))

    first_layer = 24  #manual input but fixed.
    second_layer = 14
    #print(first_layer)

    #AutoEncoder. Offline Model structure
    #weights are initialised using uniform distribution
    class AE(nn.Module):
        def __init__(self):
            super(AE, self).__init__()

            self.fc1 = nn.Linear(dim, first_layer)
            nn.init.xavier_uniform_(self.fc1.weight)
            self.fc12 = nn.Linear(first_layer, second_layer)
            nn.init.xavier_uniform_(self.fc12.weight)
            self.fc2 = nn.Linear(second_layer, latent_dim) #bottleneck
            nn.init.xavier_uniform_(self.fc2.weight)
            self.fc3 = nn.Linear(latent_dim, second_layer)
            nn.init.xavier_uniform_(self.fc3.weight)
            self.fc32 = nn.Linear(second_layer, first_layer)
            nn.init.xavier_uniform_(self.fc32.weight)
            self.fc4 = nn.Linear(first_layer, dim)
            nn.init.xavier_uniform_(self.fc4.weight)


        def encode(self, x):
            h1 = F.leaky_relu(self.fc1(x),0.2)
            h2 = F.leaky_relu(self.fc12(h1),0.2)
            return self.fc2(h2)

        def decode(self, x):
            h3 = F.leaky_relu(self.fc3(x),0.2)
            h4 = F.leaky_relu(self.fc32(h3),0.2)
            if scaled=='Robust Scaling':
                out = self.fc4(h4)
            elif scaled=='MinMax Scaling':
                out = torch.sigmoid(self.fc4(h4))
            return out  #should you use sigmoid? Yes we need it to be in [0,1] or
            # softmax ensures sum to 1.. no! /softplus.. maybe but results can be greater than [0,1]?

        def forward(self, x):
            bottleneck = self.encode(x.view(-1, dim))
            return bottleneck, self.decode(F.leaky_relu(bottleneck,0.2))

        def get_bottleneck(self,x):
            bottleneck = self.encode(x.view(-1, dim))
            return bottleneck

    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE().to(device)
    torch.manual_seed(2)
    optimizer = optim.Adadelta(model.parameters()) #weightdecay = 0. No regularisation added.
    criterion = nn.MSELoss() #comparing reconstructed value with actual value. outputs are real-valued variables

    log_interval = 10
    training_losses=[]
    def train(epochs):
        model.train()
        train_loss = 0
        for epoch in range(1,epochs+1):
            for i,data in enumerate(trainloader):
                data = data.to(device)
                optimizer.zero_grad()
                btlneck, recon_batch= model(data.float())
                loss = criterion(recon_batch,data.float())
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if epoch % log_interval == 0:
                print('epoch: {}/{} : Loss: {}'.format(epoch+1,epochs,loss.item()))

        print('====> Epoch: {} Average train loss: {:.4f}'.format(
              epoch, train_loss / len(trainloader.dataset)))
        #training_losses.append(train_loss / len(trainloader.dataset))
        training_losses.append(train_loss / len(trainloader))

    torch.manual_seed(2)
    t0 = time.time()
    epochs=400
    train(epochs)
    print('training time: {} seconds'.format(time.time() - t0))
    #sys.exit()
    #torch.save(model.state_dict(), 'AE1.pth')

    #model = AE()
    #model.load_state_dict(torch.load('AE1.pth'))

    ##################################
    #test on training data
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for i,inputs in enumerate(trainloader_all):
            inputs = inputs.to(device)
            btlneck, recon_batch = model(inputs.float())
            scores_train = torch.sum((inputs - recon_batch) ** 2, dim=1)

    print('AE test time = {}'.format(time.time() - start_time))

    #test the offline model
    idx_label_score = []
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for i,inputs in enumerate(testloader):
            inputs = inputs.to(device)
            btlneck, recon_batch = model(inputs.float())
            scores = torch.sum((inputs - recon_batch) ** 2, dim=1)

    print('AE test time = {}'.format(time.time() - start_time))

    return model,np.array(scores), np.array(scores_train)


scaling='MinMax Scaling' #'Robust Scaling'
Normalized_training_df,Normalized_testing_df = Preprocess(norm_data_enc,test_data_enc,method=scaling)
model,scores,scores_train = TrainAE(Normalized_training_df,Normalized_testing_df,scaled=scaling)


#rough check. should be all true
print(np.mean(scores)>np.mean(scores_train))

test_data['scores'] = scores

df_sorted = test_data.sort_values('DateTime').reset_index(drop=True)

df_sorted['log_scores'] = np.log(df_sorted['scores'])
df_sorted['log_scores'] = df_sorted['log_scores'] + np.abs(np.min(df_sorted['log_scores']))

df_sorted.to_csv('scored_test2.csv')

#ranking of features for latency anomalies. Not very good cause the 'latency' was introduced to detect
cols = Normalized_training_df.columns
feature_sel_df,_ = Preprocess(test_data_enc,norm_data_enc,method='MinMax Scaling')
from sklearn.feature_selection import chi2, mutual_info_classif

chi,pval = chi2(feature_sel_df[cols][:688],labels_latency[:688])
minfo = mutual_info_classif(feature_sel_df[cols][:688],labels_latency[:688])
df_rank_latency = pd.DataFrame()
df_rank_latency['features'] = cols
df_rank_latency['chi'] = chi
df_rank_latency['minfo'] = minfo
df_rank_latency = df_rank_latency.sort_values('minfo').reset_index(drop=True)

#make sure latency anomalies are not detectable here.
labels_battery = []
for i in range(len(df_sorted)):
    if df_sorted.loc[i, 'log_scores'] > 1.5:
        labels_battery.append(1)
    else:
        labels_battery.append(0)

#len(added_norm) is 298
chi,pval = chi2(feature_sel_df[cols][688+len(added_norm):],labels_battery[688+len(added_norm):])
minfo = mutual_info_classif(feature_sel_df[cols][688+len(added_norm):],labels_battery[688+len(added_norm):])
df_rank_batt = pd.DataFrame()
df_rank_batt['features'] = cols
df_rank_batt['chi'] = chi
df_rank_batt['minfo'] = minfo
#df_rank_batt = df_rank_batt.sort_values('chi').reset_index(drop=True)
df_rank_batt = df_rank_batt.sort_values('minfo').reset_index(drop=True)

####################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import matplotlib
#folder = '/home/nuc/Desktop/PythonProgs/FlowRecorder/Duncan_codes/'
def ConvertToDateTime(df):
    d_dtime_list = []
    d_time_list = []
    for i in range(len(df)):
        d_dtime_list.append(datetime.datetime.strptime(df['datetime'][i],"%d/%m/%Y %H:%M:%S"))
        d_time_list.append(datetime.datetime.strptime(df['datetime'][i], "%d/%m/%Y %H:%M:%S").time())
    df['DateTime'] = d_dtime_list
    df['Time'] = d_time_list
    return df
df_sorted = pd.read_csv('scored_test2.csv')
df_sorted = ConvertToDateTime(df_sorted)
df_sorted['baseline'] = 1.5


plt.clf()
hours = mdates.HourLocator(interval=6)   # every year
days = mdates.DayLocator()  # every month
days_fmt = mdates.DateFormatter('%d %b')

fig = plt.figure(figsize=(24,8))
spec = gridspec.GridSpec(ncols=2, nrows=2,width_ratios=[2, 6])
r1 = df_sorted.loc[688:688+297]
r2 = df_sorted.loc[688+298:]

ax00 = fig.add_subplot(spec[0])
ax00.plot('DateTime','scores',data=r1, marker='x', markerfacecolor='black', color='red', linewidth=1)
#ax00.plot('DateTime','baseline', data=r1,color='blue',label='Threshold for usual')
locator = mdates.AutoDateLocator(minticks=2, maxticks=3)
formatter = mdates.ConciseDateFormatter(locator)
ax00.xaxis.set_major_locator(locator)
ax00.xaxis.set_major_formatter(days_fmt)
ax00.set_ylim(0,5)
ax00.format_xdata = mdates.DateFormatter('%d %b')
ax00.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax00.set_title('Anomaly Scores from AE during normal operations',loc='left')
ax00.set_xlabel('Date')
ax00.set_ylabel('Anomaly Score')
ax00.grid(True)

ax01 = fig.add_subplot(spec[1])
ax01.plot('DateTime','scores',data=r2,marker='x', markerfacecolor='black', color='red', linewidth=1, label='Anomaly Scores')
ax01.plot('DateTime','baseline', data=r2,color='blue',label='Threshold for usual')
ax01.xaxis.set_major_locator(days)
ax01.xaxis.set_major_formatter(days_fmt)
ax01.xaxis.set_minor_locator(hours)
ax01.set_ylim(0,5)
ax01.format_xdata = mdates.DateFormatter('%d %b')
ax01.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax01.set_xlabel('Date')
ax01.set_ylabel('Anomaly Score')
ax01.set_title('Anomaly Scores from AE during unexpected behaviour')
ax01.legend()
ax01.grid(True)

ax10 = fig.add_subplot(spec[2])
ax10.plot('DateTime','battsV',data=r1,marker='x', markerfacecolor='black', color='green', linewidth=1 )
locator = mdates.AutoDateLocator(minticks=2, maxticks=3)
formatter = mdates.ConciseDateFormatter(locator)
ax10.xaxis.set_major_locator(locator)
ax10.xaxis.set_major_formatter(days_fmt)
ax10.format_xdata = mdates.DateFormatter('%d %b')
ax10.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax10.set_title('Observed Battery Voltage (Usual)',loc='left')
ax10.set_xlabel('Date')
ax10.set_ylabel('Battery Voltage')
ax10.grid(True)


ax11 = fig.add_subplot(spec[3])
ax11.plot('DateTime','battsV',data=r2,marker='x', markerfacecolor='black', color='green', linewidth=1 )
ax11.xaxis.set_major_locator(days)
ax11.xaxis.set_major_formatter(days_fmt)
ax11.xaxis.set_minor_locator(hours)
ax11.format_xdata = mdates.DateFormatter('%d %b')
ax11.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax11.set_title('Observed Battery Voltage (Unusual)')
ax11.set_xlabel('Date')
ax11.set_ylabel('Battery Voltage')
ax11.grid(True)

con = ConnectionPatch(xyA=(r1['DateTime'][688],1.5), xyB=(r2['DateTime'][2392],1.5),
                      coordsA="data", coordsB="data",
                      axesA=ax00, axesB=ax01, arrowstyle="-",color='blue')

ax01.add_artist(con)

plt.savefig('batts_norm_anom2.png')
##################################################################3

plt.clf()
hours = mdates.HourLocator(interval=1)   # every year
#days = mdates.DayLocator()  # every month
hours_fmt = mdates.DateFormatter('%H:%M')
fig = plt.figure(figsize=(24,8))
spec = gridspec.GridSpec(ncols=1, nrows=2)

df_sorted['log_base'] = np.log(1.5)+ np.abs(np.min(np.log(df_sorted['scores'])))
r3 = df_sorted.loc[:687]
ax01 = fig.add_subplot(spec[0])
ax01.plot('DateTime','log_scores',data=r3, marker='x', markerfacecolor='black', color='red', linewidth=1,label='Log Anomaly Scores')
ax01.plot('DateTime','log_base', data=r3,color='blue',label='Log threshold for usual')
ax01.xaxis.set_major_locator(hours)
ax01.xaxis.set_major_formatter(hours_fmt)
#ax01.xaxis.set_minor_locator(hours)
ax01.set_ylim(0,10.5)
ax01.format_xdata = mdates.DateFormatter('%H:%M')
ax01.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax01.set_xlabel('Time')
ax01.set_ylabel('Log Anomaly Score')
ax01.set_title('Anomaly Scores from AE during unexpected behaviour')
ax01.legend()
ax01.grid(True)

ax11 = fig.add_subplot(spec[1])
ax11.plot('DateTime','latency',data=r3, marker='.', markerfacecolor='black', color='green', linewidth=1,label='RTT')
ax11.xaxis.set_major_locator(hours)
ax11.xaxis.set_major_formatter(hours_fmt)
#ax01.xaxis.set_minor_locator(hours)
ax11.format_xdata = mdates.DateFormatter('%H:%M')
ax11.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax11.set_title('Observed RTT')
ax11.set_xlabel('Time')
ax11.set_ylabel('RTT')
ax11.grid(True)

plt.savefig('scores_latency2.png')




















"""
np.random.seed(10)
# use pca to convert binary columns to numerical.
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
labels = test_data_enc['label']
"""
"""

plt.clf()
plt.plot(norm_data2['battsV'][:112], marker='.', markerfacecolor='black', color='green', linewidth=1)
plt.show()

plt.clf()
plt.plot(norm_data2['DateTime'][114:264],norm_data2['battsV'][114:264], marker='.', markerfacecolor='black', color='green', linewidth=1)
plt.show()

plt.clf()
plt.plot(norm_data2['DateTime'][114:401],norm_data2['battsI'][114:401], marker='.', markerfacecolor='black', color='green', linewidth=1)
plt.show()

plt.clf()
plt.plot(norm_data_full['DateTime'][114:396],norm_data_full['arrayI'][114:396], marker='.', markerfacecolor='black', color='green', linewidth=1)
plt.show()
"""

#why multivariate time series will not work? it has strict assumptions that the data are temporally correlated
#and multivariate time series is very complicated and not a clear solution. Can you assume that the next data point depends on the previous.
#also what if your data is not in exact intervals.
#there is in fact a range of values that could occur during each hour.
#if indeed there should be an upward or downward pattern in a variable, you can input a second order change variable between two rows.

#need to ensure that for whatever variables measured, the whole range of possible normal is captured. for eg all possible behaviour for normal operations
#under each weather scenario. If not, the machine will throw out a new weather situation as an anomaly. If a scenario is missed, it can be updated
#later with new data.

#recorded in 5 min intervals but a model is built for each hour as it is the most tolerance we have for detecting anomalous behaviour.
#one or two data anomalous points in a 5 min interval may not pose a significant threat but if most of the points in the hour is anomalous, we have a problem
#the normal operations for each time period is different from other times. for eg day time and night time. Training data using only day time for anomaly
#detection at night will fail. Putting all the data together runs the risk of missing situations where night time values are present during the day!

#we can also build it over 2 hour slots and have an overlap.

#choice of model! #don't need complicated recurrent nn that can capture timeseries.
#nn is necessary because it does not prepose structures in the data. eg timeseries does -> linear. and dimensionality.
#Autoencoder cause many variables.

#don't just throw it data to let it learn. then you can't interpret the results
#the more you help the machine learn, the more the machine can help you.
