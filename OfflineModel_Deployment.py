#cleaning data

import pandas as pd
import numpy as np
from utility_functions import CheckUnique, CreateBinary, encode_onehot, Check_same_values
import datetime
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import time
import random

folder = '/home/nuc/Desktop/PythonProgs/FlowRecorder/DuncanData/Full_Data/'

norm_data = pd.read_csv(folder+'normal_data.csv')  #for training. Not needed for deployment
test_data = pd.read_csv(folder+'anomaly_data_latency.csv')  #for testing. Not needed for deployment

norm_data2 = pd.read_csv(folder+'normal_data2.csv')  #for training. Not needed for deployment
test_data2 = pd.read_csv(folder+'anomaly_data_battery.csv') #for testing. Not needed for deployment

def CreateChangeFeature(df,cols,per=1):
    """ change in variable. next row - previous row
    cols = list of columns
    will lose the first row of the data"""
    for col in cols:
        df['ch_' + col] = df[col].diff(periods=per)
    df = df.drop(labels=0,axis=0)
    return df

#function creates new change variable on some of the existing variables. Required during deployment for preprocessing.
norm_data = CreateChangeFeature(norm_data,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
norm_data2 = CreateChangeFeature(norm_data2,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
test_data = CreateChangeFeature(test_data,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])
test_data2 = CreateChangeFeature(test_data2,['battsV', 'battsSensedV','arrayV', 'arrayI','hsTemp', 'rtsTemp', 'outPower', 'inPower'])

#normal operations data taken. those that have a few discrepancies are removed for training.
df = pd.concat([norm_data2[:1300],norm_data2[1600:3200],norm_data2[3500:3800],norm_data2[4400:5100],norm_data2[5600:5900],norm_data2[7100:8400],norm_data2[10200:10800],norm_data2[11800:12500]]).reset_index(drop=True)
df2 = pd.concat([test_data2,norm_data2[6050:6350]]).reset_index(drop=True)

norm_data_full = norm_data.append(df,ignore_index=True).reset_index(drop=True)
#test_data_full = test_data.append(test_data2,ignore_index=True).reset_index(drop=True)
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
added_norm = test_data_full[-298:]['DateTime'] #removed 2 bad points

#still checking on 'signal', 'TXLatency'
#'TXRate', 'RXRate' have some problems in the test file. Removed for now. (check!?!)
#List of numerical attributes in the data. Names need to match the dataset
#try removing 'battsV', 'battsSensedV', 'latency'
numerical = ['battsI', 'arrayV', 'arrayI',
       'v_target', 'hsTemp', 'rtsTemp', 'outPower', 'inPower', 'sweep_pmax',
       'sweep_vmp', 'sweep_voc', 'minVb_daily', 'maxVb_daily', 'ahc_daily',
       'whc_daily', 'minTb_daily', 'maxTb_daily', 'weather_temp', 'weather_wind',
        'TXCapacity', 'RXCapacity','signal', 'SNR',
        'NoiseFloor', 'ch_arrayV','battsV', 'battsSensedV','ch_battsV', 'ch_battsSensedV',
       'ch_arrayI', 'ch_hsTemp', 'ch_rtsTemp', 'ch_outPower', 'ch_inPower','TXLatency','latency']

#'flags_daily' is removed for now
#list of nominal variables in the data
nominal = ['statenum','weather_long']

#list of indicator variables which have a fixed value for data #removed 'freq'
fixed_cols = ['dipswitches','CCQ'] #fixed value. anomaly if changes

#convert datatype in nominal variables to string if they are not already
for col in nominal:
    norm_data_full[col] = norm_data_full[col].astype(str)
    test_data_full[col] = test_data_full[col].astype(str)


#remove rows with latency above 10 and TXLatency above 10. Latency values above 10 are anomalies that we want to detect.
norm_data_full = norm_data_full.drop(norm_data_full[norm_data_full['latency']>10].index).reset_index(drop=True)
norm_data_full = norm_data_full.drop(norm_data_full[norm_data_full['TXLatency']>10].index).reset_index(drop=True)

#label anomaly data
test_data_full['label'] = 0
#test_data_full['type']='normal'
test_data_full.loc[test_data_full[test_data_full['latency']>10].index,'label']=1
labels_latency = test_data_full['label']
#test_data_full.loc[test_data_full[test_data_full['latency']>10].index,'type']='latency'

#test_data_full.loc[test_data_full[test_data_full['latency']>10].index,'label']=1
#test_data_full.loc[test_data_full[test_data_full['???']>10].index,'type']='battery'

#create binary feature for fixed values
CheckUnique(norm_data_full,fixed_cols) #check for training.

vals = {}
for col in fixed_cols:
    vals[col] = norm_data_full[col].unique()[0]

norm_data = CreateBinary(norm_data_full,fixed_cols,vals) #required in preprocessing of data during deployment
test_data = CreateBinary(test_data_full,fixed_cols,vals)

#check normal data in test dataset
#des_norm_train = norm_data.describe()
#des_norm_test = test_data[test_data['label']==0].describe()
#des_diff = des_norm_train - des_norm_test #numbers should be small for all columns
#print(des_diff)
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

#operation needed during deployment.
test_data = Split_by_Time(test_data,6,9)
test_data = Split_by_Time(test_data,9,11)
test_data = Split_by_Time(test_data,11,13)
test_data = Split_by_Time(test_data,13,15)
test_data = Split_by_Time(test_data,15,18)
test_data = Split_by_Time(test_data,18,21)

##############################################

#one hot encode to transform nominal features to binary. required for deployment. part of preprocessing.
norm_data_enc = encode_onehot(norm_data, nominal)
test_data_enc = encode_onehot(test_data, nominal)

Check_same_values(norm_data,test_data,norm_data_enc,test_data_enc,colms=nominal)

def Check_Columns(train_enc_columns,test_enc):
    #train, test are training and testing dataset, train_enc, test_enc are after encoded
    #to check whether encoded train and test set consist of same columns.
    #this function assumes that the training data has a fullset of columns in comparison to Check_same_values function
    for col in train_enc_columns:
        if col not in set(test_enc.columns):
            test_enc[col] = 0

#Check_Columns(norm_data_enc.columns,test_data_enc)

#list of binary variables after one hot encoding
binary = list(norm_data_enc.columns[list(norm_data_enc.columns).index('Time')+1:])

##########################################################################################################################3
#preprocess
def PCA_compress(norm_train_df, test_enc, cols):
    """
    #use pca to reduce dimensions of binary variables. cols is a list of binary variables
    Dimensionality is not reduced in this function
    Then the data is normalised to range 0-1"""
    tot = len(cols)
    while True:
        pca = PCA(n_components=tot)
        pca.fit(norm_train_df[cols])
        pca_bin_normal_tr = pca.transform(norm_train_df[cols])
        if len(set(pca_bin_normal_tr[:,tot-1]))!=1:
            break
        tot-=1

    print('{}% of the variance is explained.'.format(sum(pca.explained_variance_ratio_)*100))
    #99.992% variance explained
    collist = []
    for di in range(tot):
        collist.append('D{}'.format(di+1))

    #scale pca dimensions to range (0,1)
    scaler = MinMaxScaler()   #could use sth else? #needs to be within 0-1.
    scaler.fit(pca_bin_normal_tr)
    scaled_pca_bin_normal_tr = scaler.transform(pca_bin_normal_tr)
    scaled_pca_bin_normal_tr  = pd.DataFrame(scaled_pca_bin_normal_tr , columns=collist)

    pca_bin_ts = pca.transform(test_enc[cols])
    scaled_pca_bin_ts = scaler.transform(pca_bin_ts)
    scaled_pca_bin_ts  = pd.DataFrame(scaled_pca_bin_ts,columns=collist)

    return scaled_pca_bin_normal_tr, scaled_pca_bin_ts, pca, scaler, tot

def PCA_compress_with_transformer(test_enc, cols,pca_transformer,pca_scaler,tot):
    """
    same as above functions but with the transfoermer and scaler. no need for training data.
    #use pca to reduce dimensions of binary variables. cols is a list of binary variables
    Dimensionality is not reduced in this function
    Then the data is normalised to range 0-1"""

    #99.992% variance explained
    collist = []
    for di in range(tot):
        collist.append('D{}'.format(di+1))

    #scale pca dimensions to range (0,1)
    scaler = pca_scaler
    pca = pca_transformer
    pca_bin_ts = pca.transform(test_enc[cols])
    scaled_pca_bin_ts = scaler.transform(pca_bin_ts)
    scaled_pca_bin_ts  = pd.DataFrame(scaled_pca_bin_ts,columns=collist)

    return scaled_pca_bin_ts

def Preprocess(norm_data_enc,test_data_enc,method='Robust Scaling'):

    np.random.seed(10)
    #use pca to convert binary columns to numerical.

    if method=='Robust Scaling':
        scaler = RobustScaler().fit(norm_data_enc[numerical])
    elif method=='MinMax Scaling':
        scaler = MinMaxScaler().fit(norm_data_enc[numerical])

    Normalized_training_df = pd.DataFrame(scaler.transform(norm_data_enc[numerical]),columns=numerical)
    Normalized_testing_df = pd.DataFrame(scaler.transform(test_data_enc[numerical]),columns=numerical)

    return Normalized_training_df,Normalized_testing_df,scaler


###########################################################################################################################

#AutoEncoder. Offline Model structure
#weights are initialised using uniform distribution
def InitialiseModel(dim,first_layer,second_layer,latent_dim,scaled):
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
    return model


def TrainAE(Normalized_training_df,scaled='Robust Scaling'):

    random.seed(30)
    np.random.seed(30)
    print(len(Normalized_training_df))

    #convert data into tensor
    dataset = torch.tensor(Normalized_training_df.values)

    batchsize = int(0.05*len(Normalized_training_df))  #(1/20 th of the training dataset)
    #batchsize = 500

    trainloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1) #load with respect to batch size
    trainloader_all = DataLoader(dataset, len(Normalized_training_df), shuffle=True, num_workers=1) #load all data for score calculation

    dim =np.shape(Normalized_training_df)[1]
    print(dim)
    latent_dim = int(round(np.sqrt(dim)+1))

    first_layer = 24  #manual input but fixed.
    second_layer = 14
    #print(first_layer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InitialiseModel(dim,first_layer,second_layer,latent_dim,scaled)

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


    return model, np.array(scores_train)


scaling='MinMax Scaling' #'Robust Scaling'
scaled_pca_bin_normal_tr, scaled_pca_bin_ts, pca_transformer, pca_scaler,tot = PCA_compress(norm_data_enc, test_data_enc, cols=binary) #preprocessing on binary variables
Normalized_training_df,Normalized_testing_df,scaler = Preprocess(norm_data_enc,test_data_enc,method=scaling) #preprocessing on numerical variables
Normalized_training_df = pd.concat([Normalized_training_df, scaled_pca_bin_normal_tr], axis=1)
Normalized_testing_df = pd.concat([Normalized_testing_df, scaled_pca_bin_ts], axis=1)
model,scores_train = TrainAE(Normalized_training_df,scaled=scaling)

torch.save(model.state_dict(), 'AE.pth')

#####################################


###load model from pth file. the structure must be the same. need to pass in same parameters as initialise model function.
def LoadModel(dim,first_layer,second_layer,latent_dim,scaling,fname='AE.pth'):
    nmodel = InitialiseModel(dim,first_layer,second_layer,latent_dim,scaling)
    nmodel.load_state_dict(torch.load(fname))
    return nmodel

first_layer = 24  # manual input but fixed.
second_layer = 14
dim = np.shape(Normalized_training_df)[1]
print(dim)
latent_dim = int(round(np.sqrt(dim) + 1))


#######################################################
#deploy this model
model_for_deployment = LoadModel(dim,first_layer,second_layer,latent_dim,scaling,fname='AE.pth')

def ScoreLiveData(data,model):
    """data is set of data in a time window to evaluate. can be 1 data point.
    this data must be preprocessed and in the same format as the data used to train the model
    model is the deployed model
    returns anomaly score of each data point in the time window"""
    torchdata = torch.tensor(data.values)
    testloader = DataLoader(torchdata, len(data), shuffle=False, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for i,inputs in enumerate(testloader):
            inputs = inputs.to(device)
            btlneck, recon_batch = model(inputs.float())
            scores = torch.sum((inputs - recon_batch) ** 2, dim=1)

    print('AE test time = {}'.format(time.time() - start_time))

    return np.array(scores)



def PreprocessLiveData(data,all_encoded_columns):
    """data is a pandas dataframe with at least 2 rows because it needs to calculate change for certain variables
    all_encoded_columns is the columns of the data used to train the model after onehot encoding but before normalizing
    """
    #maybe a better way to create the change feature because, you lose one data point by running this function as you process live data by batch
    #not advisable to process each data point individually
    datas = CreateChangeFeature(data, ['battsV', 'battsSensedV', 'arrayV', 'arrayI', 'hsTemp', 'rtsTemp', 'outPower','inPower'])

    datas = datas.drop(datas[datas['battsI'] > 120].index, axis=0).reset_index(drop=True) #some measurement errors. we drop this data point. a better way available?
    datas = ConvertToDateTime(datas) #input new column for datetime

    for col in nominal:
        #nominal is list of columns with nominal variables.
        datas[col] = datas[col].astype(str)

    datas = CreateBinary(datas, fixed_cols, vals) #fixed cols is variables with the same value and the vals is the dictionary of them

    datas = Split_by_Time(datas , 6, 9) #assigns a new variable if the data point is within the time frame of 6 am to 9am.
    datas  = Split_by_Time(datas , 9, 11)
    datas  = Split_by_Time(datas , 11, 13)
    datas  = Split_by_Time(datas , 13, 15)
    datas  = Split_by_Time(datas , 15, 18)
    datas  = Split_by_Time(datas , 18, 21)

    data_enc = encode_onehot(datas, nominal)
    Check_Columns(all_encoded_columns, data_enc)

    scaled_pca_bin_ts = PCA_compress_with_transformer(data_enc, binary, pca_transformer, pca_scaler,tot)
    #binary is the list of binary columns, pca_transformer and pca_scaler are determined while preprocessing training data

    Normalized_data_df = pd.DataFrame(scaler.transform(data_enc[numerical]), columns=numerical)
    #scaler is the scaler for numerical data determined while preprocessing training data

    Normalized_data_df = pd.concat([Normalized_data_df, scaled_pca_bin_ts], axis=1)

    return Normalized_data_df


#eg: taking the whole test data together.
#Normalized_data_df = PreprocessLiveData(test_data,norm_data_enc.columns)
#scores = ScoreLiveData(Normalized_data_df,model_for_deployment)