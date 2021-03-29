#utility funcs
from itertools import combinations
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import roc_curve
from scipy import stats



def CheckSimilarity(numerical_cols,data):
    #data is pandas dataframe, numerical_cols is list
    #checks similarity between two columns
    for combo in combinations(range(0,len(numerical_cols)), 2): #Choose any two numerical columns
        f1 = numerical_cols[combo[0]]
        f2 = numerical_cols[combo[1]]
        if (data[f1]==data[f2]).all():
            print('{} and {} are exactly the same'.format(f1,f2))

def CheckUnique(data,numerical):
    # data is pandas dataframe, numerical is list of columns.
    # function checks if a col has more than one value or only a unique value.
    # unique value does not provide any information to machine learning algo unless the uniqueness is guranteed
    for col in numerical:
        if len(data[col].unique()) == 1:
            print('{} has only 1 unique value of {}'.format(col, data[col].unique()))

def CreateBinary(data,fixed_cols,vals):
    #data is pd dataframe, fixed cols is list of columns which are supposed to have fixed values
    #vals is a dict with key as the col name and value as the fixed value
    #convert the fixed value column value to 1 (normalisation)
    for col in fixed_cols:
        colf = col+'f'
        data[colf] = 0
        data.loc[data[data[col]==vals[col]].index,colf]=1
    return data

def CheckNegativeVal(numerical_cols,data):
    # data is pandas dataframe, numerical_cols is list
    #function checks if any value in the column is negative.
    for col in numerical_cols:
        if min(data[col]) < 0:
            count = 0
            for row in data.index:
                if data[col][row] < 0:
                    count += 1
            print('{} has {} negative values'.format(col, count))


def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()

    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index

    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def Check_same_values(train,test,train_enc,test_enc,colms=['service','state']):
    #train, test are training and testing dataset, train_enc, test_enc are after encoded
    #to check whether encoded train and test set consist of same columns.
    for col in colms:
        not_in_test = set(train[col]) - set(train[col]).intersection(set(test[col]))
        not_in_train = set(test[col]) - set(train[col]).intersection(set(test[col]))
        print(not_in_test)
        print(not_in_train)
        if not_in_train:
            for item in not_in_train:
                col_name = col+'={}'.format(item)
                train_enc[col_name]=0
        if not_in_test:
            for item in not_in_test:
                col_name = col+'={}'.format(item)
                test_enc[col_name]=0


def CreateChangeFeature(df,cols,per=1):
    """ change in variable. next row - previous row
    cols = list of columns
    will lose the first row of the data"""
    for col in cols:
        df['ch_' + col] = df[col].diff(periods=per)
    df = df.drop(labels=0,axis=0)
    return df

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

    return scaled_pca_bin_normal_tr, scaled_pca_bin_ts

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters:
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns : value with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0]

def classify_pts(labels,scores):
    thres = Find_Optimal_Cutoff(labels, scores)
    print(thres)
    pred = []
    for sc in scores:
        if sc<thres:
            pred.append(0)
        else:
            pred.append(1)

    df_res = pd.DataFrame(columns=['actual','pred'])
    df_res['actual']=labels.reshape(-1)
    df_res['pred']=pred
    return df_res

def evaluate_rate(df_res):
    try:
        fpr = len(df_res[(df_res['actual']==0) & (df_res['pred']==1)])/len(df_res[df_res['actual']==0])
        tpr = len(df_res[(df_res['actual']==1) & (df_res['pred']==1)])/len(df_res[(df_res['actual']==1)])
        accuracy = (len(df_res[(df_res['actual']==1) & (df_res['pred']==1)])+
                    len(df_res[(df_res['actual']==0) & (df_res['pred']==0)]))/len(df_res)

        precision = len(df_res[(df_res['actual']==1) & (df_res['pred']==1)])/len(df_res[df_res['pred']==1])
        f1 = 2*(precision*tpr)/(tpr+precision)
    except ZeroDivisionError:
        print('all labels are the same')
        fpr, tpr, accuracy, precision, f1 = -1,-1,-1,-1,-1

    return (fpr,tpr,accuracy,precision,f1)

#######################
#for online training of behaviour based model AE-OCSVM
def select_data_online_ocsvm(ocsvm_df,ocsvm,df_for_online,retainsv=True):
    #training data for ocsvm, the trained ocsvm, df_online: the new data for training.
    if retainsv:
        support_vec = ocsvm_df.loc[ocsvm.support_]
        new_ocsvm_df = pd.concat([support_vec,df_for_online]).reset_index(drop=True)
    else:
        new_ocsvm_df = df_for_online
    return new_ocsvm_df

def Select_scores(tr_scores,per = 85):
    iqr = stats.iqr(tr_scores)
    lq = 0
    uq = np.percentile(tr_scores,per) + 0*iqr  #configurable parameter to determine how much of the batch has different profiles
    qr_arr = np.where((tr_scores>lq) & (tr_scores<uq))

    return tr_scores[qr_arr]

#########################
#for online training of binary classification model Rad-NN-SVM

def ConfidenceMeasure(p,label,NormalNeigh,AnomalyNeigh,k=10):
    #Database is the df used to train the Neighbors
    #p is the testing data point, with its predicted label
    #NormalNeigh and ANomalyNeigh are NearestNeighbor classifiers trained respectively
    distances,inx = NormalNeigh.kneighbors(p,n_neighbors=k)
    Total_norm_dist = sum(list(distances[0]))
    distances,inx = AnomalyNeigh.kneighbors(p,n_neighbors=k)
    Total_anom_dist = sum(list(distances[0]))
    if label == 0:
        ratio = Total_norm_dist/Total_anom_dist
    else:
        ratio = Total_anom_dist/Total_norm_dist
    return (1-ratio)


def PredictConfidence(p, rneigh, NormalNeigh, AnomalyNeigh, svm, thres=0.8):
    label = rneigh.predict(p)
    if label[0] != -1: #when rad_nn is unable to predict
        confi = ConfidenceMeasure(p, label, NormalNeigh, AnomalyNeigh, k=10)
        if confi < thres:
            new_label = svm.predict(p)
            if new_label[0] == label[0]:
                confi = max(confi, max(svm.predict_proba(p)[0]))
            else:
                if max(svm.predict_proba(p)[0]) > confi:
                    label = new_label
                    confi = max(svm.predict_proba(p)[0])
                    # Add indicator to denote how many points were classified by svm
    else:
        label = svm.predict(p)
        confi = max(svm.predict_proba(p)[0])
    return (label[0], confi)


def Assign_to_log(flow, label, confi, HC_norm_df, HC_annom_df, LC_df, uthres=0.9,
                  lthres=0.8, label_col='normal', confi_col='confidence'):
    labeled_data = flow.append(pd.Series({label_col: label}))
    labeled_confi_data = flow.append(pd.Series({label_col: label, confi_col: confi}))
    if (confi > uthres and label == 0):
        HC_norm_df.loc[len(HC_norm_df)] = labeled_data
    elif (confi > uthres and label == 1):
        HC_annom_df.loc[len(HC_annom_df)] = labeled_data
    elif confi < lthres:
        LC_df.loc[len(LC_df)] = labeled_confi_data
    return (HC_norm_df, HC_annom_df, LC_df)

def Assign_to_log2(flow, label, confi, HC_norm_df, HC_annom_df, LC_df, uthres=[0.9,0.8],
                  lthres=0.8, label_col='normal', confi_col='confidence'):
    #high confidence threshold for normal is uthres[0] and for anomalies is uthres[1]
    #when there is not enough anomaly data.
    labeled_data = flow.append(pd.Series({label_col: label}))
    labeled_confi_data = flow.append(pd.Series({label_col: label, confi_col: confi}))
    if (confi > uthres[0] and label == 0):
        HC_norm_df.loc[len(HC_norm_df)] = labeled_data
    elif (confi > uthres[1] and label == 1):
        HC_annom_df.loc[len(HC_annom_df)] = labeled_data
    elif confi < lthres:
        LC_df.loc[len(LC_df)] = labeled_confi_data
    return (HC_norm_df, HC_annom_df, LC_df)

