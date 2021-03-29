##binary hml
from data_clean import Normalized_training_df,Normalized_testing_df, labels
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier
from sklearn.svm import SVC
from utility_functions import evaluate_rate, PredictConfidence, Assign_to_log, Assign_to_log2
import time

Normalized_testing_df['label']=labels

#sample 50% of the anomaly data for training.
nbr_anom = int(0.5*np.sum(labels))
anomaly_training = Normalized_testing_df[Normalized_testing_df['label']==1].sample(nbr_anom,random_state=10)
normal_training = Normalized_training_df.sample(2*nbr_anom,random_state=2) #twice equivalent amount of normal data   ##????? change this
normal_training['label'] = 0

training = normal_training.append(anomaly_training,ignore_index=True).reset_index(drop=True)
testing =  Normalized_testing_df.drop(anomaly_training.index).reset_index(drop=True)

#normal_testing = Normalized_testing_df[Normalized_testing_df['label']==0].sample(nbr_anom,random_state=10)
#anomaly_testing = Normalized_testing_df[Normalized_testing_df['label']==1].drop(anomaly_training.index)

cols = Normalized_training_df.columns

#weight the features
chi,pval = chi2(training[cols],training['label'])
feat_wt = (chi - chi.min())/(chi.max()-chi.min())

#Normalized_testing_df[Normalized_testing_df<0].any()
#Normalized_testing_df = Normalized_testing_df.drop([73],axis=0).reset_index(drop=True)
#chi,pval = chi2(Normalized_testing_df[cols],Normalized_testing_df['label'])
#df_rank = pd.DataFrame()
#df_rank['features'] = cols
#df_rank['chi'] = chi
#df_rank = df_rank.sort_values('chi').reset_index(drop=True)

#determine an optimal radius looking at average distance of all the points in space
neigh = NearestNeighbors(n_neighbors=10, radius=1.0,metric='wminkowski',metric_params={'w': feat_wt})
neigh.fit(training[cols],training['label']) #This is an in itial radius for the model.

dist = []
de = (len(training)-1)
for row in training.index:
    distances,inx = neigh.kneighbors([training[cols].loc[row]],n_neighbors = len(training))
    Av_dist = sum(list(distances[0]))/de
    dist.append(Av_dist)

rad = np.percentile(np.array(dist),50)
#print('We choose the 50th percentile because there are many outliers and the distribution is fat tailed.')
print('radius is  {}'.format(rad))

#train radius nearest neighbours
Radneigh = RadiusNeighborsClassifier(radius=rad, weights='distance',metric='wminkowski',
                                  outlier_label=-1,metric_params={'w': feat_wt})
Radneigh.fit(training[cols],training['label'])


#to measure confidence of Rad-NN
k=10
NormalNeigh=NearestNeighbors(n_neighbors= k,metric='wminkowski',metric_params={'w': feat_wt})
AnomalyNeigh = NearestNeighbors(n_neighbors= k,metric='wminkowski',metric_params={'w': feat_wt})
NormalNeigh.fit(training[training['label']==0][cols])
AnomalyNeigh.fit(training[training['label']==1][cols])


def Select_Data(HC_norm_df, HC_annom_df, svm_df, svm):
    #select data for online training
    support_vec = svm_df.loc[svm.support_]
    new_svm_df = pd.concat([support_vec, HC_norm_df, HC_annom_df]).reset_index(drop=True)
    return new_svm_df


def Retrain_svm(svm_df, selc_col, c, gam, label_col='normal'):
    # length of df should be 1000
    new_svm = SVC(C=c, gamma=gam, probability=True)
    new_svm.fit(svm_df[selc_col], svm_df['label'])
    return new_svm

#take the first few datapoints to initialise the online model. 123 points to get some anomaly points as well.
svm = SVC(C= 100, gamma = 0.1, probability=True) #We want overfitting. Low bias and high variance to start with
init_row=123
svm.fit(testing[cols].loc[0:init_row],testing['label'].loc[0:init_row])
svm_df = testing.loc[0:init_row]

ccols = list(testing.columns)
ccols.append('confidence')

HC_norm_df = pd.DataFrame(columns = training.columns)
HC_annom_df  = pd.DataFrame(columns = training.columns)
LC_df = pd.DataFrame(columns = ccols)
predn = []
tr = 0

starttime = time.time()
for i in testing[init_row+1:].index:
    flow = testing[cols].loc[i]
    p = [flow]
    label,confi = PredictConfidence(p,Radneigh,NormalNeigh,AnomalyNeigh,svm,thres=0.9)
    predn.append(label)
    HC_norm_df,HC_annom_df,LC_df = Assign_to_log2(flow,label,confi,HC_norm_df,HC_annom_df,LC_df,
                                                 uthres = [0.97,0.91],lthres = 0.85,label_col='label')
    nbr_new_data = len(svm.support_)+len(HC_norm_df)+len(HC_annom_df)
    if nbr_new_data > 150:
        #Retrain SVM
        tr+=1
        print('svm retraining {}'.format(tr))
        svm_df = Select_Data(HC_norm_df,HC_annom_df,svm_df,svm)
        c = 100
        gam = 0.1
        svm = Retrain_svm(svm_df,cols,c,gam,label_col='normal')
        HC_norm_df = pd.DataFrame(columns = training.columns)
        HC_annom_df  = pd.DataFrame(columns = training.columns)

print ('time taken:{:.2f} mins'.format((time.time() - starttime)/60))


#svm trained using 1000 samples.Initialise svm with C=1000, gamma=0.1, 13 svm trained. 28.35 min
dfresults = pd.DataFrame()
dfresults['actual'] = testing['label'].loc[init_row+1:]
dfresults['pred']=predn

k = evaluate_rate(dfresults)
print('fpr:{:.2f}%, tpr:{:.2f}%, acc:{:.2f}%, prec:{:.2f}%, f1:{:.2f}%'.format(100*k[0],100*k[1],100*k[2],100*k[3],100*k[4]))

