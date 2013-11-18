import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from tools import medianFilter,smooth,mycat
import pywt,pickle,re,sys
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,scale
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.decomposition import PCA

def signalProcessFeatures(signal,showfeature=False):
    #signal is 2D MxN
    SHAPE = signal.shape
    signal_smooth = np.array([])
    for col in range(SHAPE[1]):
        #signal[:,col] = medianFilter(signal[:,col],k=299)
        ## TODO: why does smooth() return a signal of larger size??
        #signal[:,col] = smooth(signal[:,col],window_len=9)[:SHAPE[0]]
        ssm = smooth(signal[:,col],window_len=299)
        signal_smooth = mycat(signal_smooth,ssm,dim=1)
        #signal[:,col] = medianFilter(signal[:,col],k=9)
        #mu = signal[:,col].mean()
        #sigma = signal[:,col].std()
        #signal[:,col] = (signal[:,col]-mu)/float(sigma)
        #signal[:,col]=signal[:,col]


    #for row in range(SHAPE[0]):
        #signal[row,:]=medianFilter(signal[row,:],k=9)
        #ssm = smooth(signal[row,:],window_len=9)
        #signal_smooth = mycat(signal_smooth,ssm,dim=0)
    #signal_smooth=signal
    #signal_smooth = signal_smooth[:,:SHAPE[1]]

    print signal.shape
    print signal_smooth.shape
    if showfeature:
        pcolor(signal[:,:10].T)
        show()

    return signal_smooth

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

            #for each spectra, pull out the Approximation and Detail coefficients
            for spectraIdx in range(SHAPE[1]): #for each of 100 spectra
                coefs = pywt.wavedec(signal[:,spectraIdx].flatten(),pywtModel,level=level)
                fv_i = np.array([])
                for l,c in enumerate(coefs):
                    #if l>1:# or l>=3: #only approximation
                    #    break
                    fv_i=np.append(fv_i,c)#[:min(len(c),5)]) #5
                fv = mycat(fv,fv_i,dim=1)

            #extract the reconstructed signal
            #for spectraIndex in range(SHAPE[1]):
            #    wavelet = pywt.Wavelet(pywtModel)
            #    phi, psi, x = wavelet.wavefun(level=level)

        
        if feature=='mean': #add sliding window mean?
            fv = np.mean(signal,axis=0)

        elif feature == 'std':
            fv = np.std(signal,axis=0)

        FV = mycat(FV,fv,dim=0)

    return FV

if __name__=='__main__':

    #python2.7 main.py True sample_data/cancer_train.data sample_data/cancer_train.labels svm.pkl
    narg = len(sys.argv)

    if narg>4 and narg <7:
        spectraFile = sys.argv[1]
        numberOfSpectra = int(sys.argv[2])
        retrain = sys.argv[3]=='True'
        pickleName = sys.argv[4] #name to load from or dump to
        if narg==6:
            spectraLabels = sys.argv[5]

    else:
        raise Exception('Must have 4 or 5 inputs: python main.py\
        <spectraFile>\
        <number of spectra per experiment>\
        <doRetrain>\
        <pickleName>\
        <spectraLabels>')


    msdata_flat=np.fromfile(spectraFile,sep='\t')
    print len(msdata_flat)
    print len(msdata_flat)/int(numberOfSpectra)
    MSSHAPE = [numberOfSpectra,len(msdata_flat)/numberOfSpectra]

    msTrain = np.reshape(msdata_flat,MSSHAPE)

    msTrainClean = signalProcessFeatures(msTrain)
    #msTrainClean = msTrain
    trainFV = features(msTrainClean,featurelist=['mean','std','pywt.haar.4','pywt.db2.4','pywt.coif1.4'])
                                                 # 'pywt.db2.2','pywt.db2.4','pywt.db2.8','pywt.db2.16'])#,\
                                                 #'pywt.db2.400','pywt.db2.1000'])
    #mu = trainFV.mean()
    #sigma = trainFV.std()
    #trainFV_n = (trainFV-mu)/float(sigma)
    trainFV_n = scale(trainFV)
    #scaler = StandardScaler()
    #trainFV_n = scaler.transform(trainFV)

    #mu = trainFV.mean(axis=1)
    #sigma = trainFV.std(axis=1)
    #mu = np.tile(mu[:,np.newaxis],(1,MSSHAPE[1]))
    #sigma = np.tile(sigma[:,np.newaxis],(1,MSSHAPE[1]))
    #trainFV_n = (trainFV-mu)/sigma

    if retrain:
        msTrainLabel = np.fromfile(spectraLabels,sep='\t')

        pca = PCA(n_components=2)
        fv_pca = pca.fit_transform(trainFV_n.T)

        marker={1:'*r',-1:'ob'}
        figure()
        for i,tl in enumerate(msTrainLabel):
            plot(fv_pca[i,0],fv_pca[i,1],marker[tl],markersize=16)
        show()

        svc_params={'C': np.logspace(-1,2,4),
                    'gamma': np.logspace(-4,0,5)
                    }
        gs_svc = GridSearchCV(SVC(),svc_params,scoring='roc_auc',n_jobs=-1)
        gs_svc.fit(trainFV_n.T,msTrainLabel,cv=LeaveOneOut(len(msTrainLabel)))
        
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
        
        pickle.dump(svmmodel,open('output/'+pickleName,'wb'))
    else:
        svmmodel = pickle.load(open('output/'+pickleName,'rb'))
        print svmmodel.predict(trainFV_n.T)-np.fromfile('sample_data/cancer_train_20.labels',sep='\t')

