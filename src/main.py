import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools import medianFilter,smooth
import pywt,pickle,re,sys
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

def signalProcessFeatures(signal,showfeature=False):
    #signal is 2D MxN
    SHAPE = signal.shape
    for col in range(SHAPE[1]):
        #signal[:,col] = medianFilter(signal[:,col],k=9)
        ## TODO: why does smooth() return a signal of larger size??
        signal[:,col] = smooth(signal[:,col],window_len=10)[:SHAPE[0]]


    if showfeature:
        pcolor(signal[:,:5].T)
        show()

    return signal

def mycat(s1,s2,dim=0):

    if dim ==0:

        if s1.ndim == 1:
            #edge case, first time appending
            if s1.shape==(0,):
                if s2.ndim==1:
                    return s2[np.newaxis,:]
                else:
                    return s2

            s1 = s1[np.newaxis,:]
        if s2.ndim == 1:
            s2 = s2[np.newaxis,:]
        if not s1.shape[1] == s2.shape[1]:
            raise Exception('Input signals are not the same shape: '+str(s1.shape)+' '+str(s2.shape))
        return np.concatenate((s1,s2),axis=0)

    elif dim==1:
        if s1.ndim == 1:
            #edge case, first time appending
            if s1.shape==(0,):
                if s2.ndim==1:
                    return s2[:,np.newaxis]
                else:
                    return s2

            s1 = s1[:,np.newaxis]
        if s2.ndim == 1:
            s2 = s2[:,np.newaxis]
        if not s1.shape[0] == s2.shape[0]:
            raise Exception('Input signals are not the same shape: '+str(s1.shape)+' '+str(s2.shape))
        return np.concatenate((s1,s2),axis=1)

    else:
        raise Exception('Invalid dimension, only accepts 0 or 1')

def features(signal,featurelist=['pywt.db1.1','pywt.db1.5','mean','std','pywt.haar.5']):
    #signal is 2D: cols are samples, rows are features

    FV = np.array([])
    SHAPE = signal.shape

    for feature in featurelist: #for each feature to extract
        fv = np.array([])

        if 'pywt' in feature:
            featurestr = re.split('\.',feature)
            pywtModel = featurestr[1] #grab feature
            level = int(featurestr[2]) #grab level

            for spectraIdx in range(SHAPE[1]): #for each of 100 spectra
                coefs = pywt.wavedec(signal[:,spectraIdx].flatten(),pywtModel,level=level)
                fv_i = np.array([])
                for l,c in enumerate(coefs):
                    if l>=2:
                        break
                    fv_i=np.append(fv_i,c[:min(len(c),5)]) #5
                fv = mycat(fv,fv_i,dim=1)
        
        if feature=='mean':
            fv = np.mean(signal,axis=0)
        elif feature == 'std':
            fv = np.std(signal,axis=0)

        FV = mycat(FV,fv,dim=0)

    return FV

if __name__=='__main__':

    #python2.7 main.py True sample_data/cancer_train.data sample_data/cancer_train.labels svm.pkl
    narg = len(sys.argv)

    if narg>2:
        retrain = sys.argv[1]=='True'
        if retrain:
            spectraFile = sys.argv[2]
            spectraLabels = sys.argv[3]
            modeltosave = sys.argv[4]
        else:
            spectraFile = sys.argv[2]
            modeltoload = sys.argv[3]
    else:
        raise Exception('Must have 2 or 3 inputs: python main.py <retrain><spectraFile><spectraLabels>')


    MSSHAPE = [10000,100]
    msdata_flat=np.fromfile(spectraFile,sep='\t')
    msTrain = np.reshape(msdata_flat,MSSHAPE)

    msTrainClean = signalProcessFeatures(msTrain)
    #msTrainClean = msTrain
    trainFV = features(msTrainClean)
    trainFV_n = (trainFV-trainFV.mean())/float(trainFV.std())
    #scaler = StandardScaler()
    #trainFV_n = scaler.transform(trainFV)

    if retrain:
        msTrainLabel = np.fromfile(spectraLabels,sep='\t')
        
        svc_params={'C': np.logspace(-1,2,4),
                    'gamma': np.logspace(-4,0,5)
                    }
        gs_svc = GridSearchCV(SVC(),svc_params,scoring='roc_auc')
        gs_svc.fit(trainFV_n.T,msTrainLabel)
        
        print trainFV_n.shape
        print '================'
        print gs_svc
        print gs_svc.best_params_
        print gs_svc.best_score_
        print gs_svc.best_estimator_
        
        print '================'
        svmmodel = gs_svc.best_estimator_
        trainPred = svmmodel.predict(trainFV_n.T)
        
        print classification_report(msTrainLabel,trainPred)
        print confusion_matrix(msTrainLabel,trainPred)
        
        pickle.dump(svmmodel,open(modeltosave,'wb'))
    else:
        svmmodel = pickle.load(open(modeltoload,'rb'))
        print svmmodel.predict(trainFV_n.T)

