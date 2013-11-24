import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from tools import medianFilter,smooth,mycat
import pywt,pickle,re,sys
from sklearn.preprocessing import StandardScaler,scale
from sklearn.metrics import classification_report,confusion_matrix
from scipy import fft,ifft
from peakDetection import peakDetection
from mlModels import *

def normalize(signal,learnNormalization=False):
    #signal is 2D MxN
    #return signal
    SHAPE = signal.shape

    if learnNormalization:
        #smooth across time for each mass bin
        mu_estimator = signal.mean(axis=1)
        sigma_estimator = signal.std(axis=1)
        pickle.dump((mu_estimator,sigma_estimator),open('output/normalizations.pkl','wb'))
    else:
        mu_estimator,sigma_estimator = pickle.load(open('output/normalizations.pkl','rb'))
    mu = np.tile(mu_estimator[:,np.newaxis],(1,SHAPE[1]))
    sigma = np.tile(sigma_estimator[:,np.newaxis],(1,SHAPE[1]))
    signal_smooth = (signal-mu)/sigma
    signal_smooth = np.log(signal_smooth)

    '''
    mu = np.tile(signal.mean(axis=0),(SHAPE[0],1))
    sigma = np.tile(signal.std(axis=0),(SHAPE[0],1))
    signal_smooth = (signal-mu)/sigma
    '''
    #signal_smooth[signal_smooth<1]=1

    signal_smooth[np.isnan(signal_smooth)]=0
    signal_smooth[np.isinf(signal_smooth)]=0

    return signal_smooth

def features(signal,featurelist=['pywt.db1.1','pywt.db1.5','mean','std','pywt.haar.5']):
    #signal is 2D: cols are samples, rows are features

    FV = np.array([])
    SHAPE = signal.shape

    for feature in featurelist: #for each feature to extract
        fv = np.array([])

        if feature == 'fullSignal':
            fv=signal

        elif feature=='autocorrelation':
            for col in range(SHAPE[1]):
                #fv_ij=np.array([])
                #for i in range(len(fToCorrelate)):
                #    for j in range(i+1,len(fToCorrelate)):
                #        fv_ij=np.append(fv_ij,np.abs(signal[i,col]-signal[j,col])/float(np.abs(signal[i,col])+np.abs(signal[j,col])))
                acor = np.correlate(signal[:,col],signal[:,col],mode='full')
                fv = mycat(fv,acor)
                #fv=mycat(fv,fv_ij)
            fv=fv.T
            fv[np.isnan(fv)]=0

        elif feature == 'topBins':
            for col in range(SHAPE[1]):
                topBins = np.argsort(signal[:,col])[-10:]
                fv=mycat(fv,topBins)
            fv=fv.T


        elif 'pywt' in feature:
            PEAKDETECT = False
            featurestr = re.split('\.',feature)
            pywtModel = featurestr[1] #grab feature
            level = int(featurestr[2]) #grab level

            #for each spectra, pull out the Approximation and Detail coefficients
            for spectraIdx in range(SHAPE[1]): #for each of 100 spectra
                #level = pywt.swt_max_level(SHAPE[1])

                coefs = pywt.swt(signal[:,spectraIdx].flatten(),pywt.Wavelet(pywtModel),level=level)


                fv_i = np.array([])
                for l,coef in enumerate(coefs): #for each level of coefficient

                    for aOrD,c in enumerate(coef): #cA,cD

                        c_fft = fft(c)[:40]
                        fv_i=np.append(fv_i,c_fft)

                        if PEAKDETECT:
                            [a,b]=peakDetection(c,.8)
                            numPeaks = 10
                            #grab coefficient location of 10 largest peaks
                            if len(a)==0:
                                f_to_append = np.zeros(numPeaks)
                            else:
                                idx_to_append = np.argsort(a[:,1])[::-1]

                                if len(idx_to_append)<numPeaks:
                                    f_to_append = np.concatenate((a[idx_to_append,0],np.zeros(numPeaks-len(idx_to_append))))
                                elif len(idx_to_append)>numPeaks:
                                    f_to_append = a[idx_to_append[:numPeaks],0]
                                else:
                                    f_to_append = a[idx_to_append,0]
                            #binarize: is a peak or not
                            f_to_append = np.zeros(len(c))
                            if len(a)>0:
                                f_to_append[a[:,0].astype(int)]=1
                            fv_i=np.append(fv_i,f_to_append)

                        #append mean and std of the detail signal
                        mu = np.mean(c)
                        sigma = np.std(c)
                        if np.isnan(sigma): sigma=0
                        fv_i=np.append(fv_i,mu)
                        fv_i=np.append(fv_i,sigma)

                fv = mycat(fv,fv_i,dim=1)

        
        elif feature=='mean': #add sliding window mean?
            fv = np.mean(signal,axis=0)

        elif feature == 'std':
            fv = np.std(signal,axis=0)

        else:
            raise Exception('Invalid feature')

        FV = mycat(FV,fv,dim=0)

    return FV

def binarizePred(X):
    zidx = X<0
    X[zidx]=-1
    X[~zidx]=1
    return X

def myScale(X,type='rowmax'):

    if type=='rowmax':
        mx = X.max(axis=1)
        X= X/np.tile(mx[:,np.newaxis],(1,X.shape[1]))
    elif type == 'matrixmax':
        X= X/float(X.max())
    elif type == 'scikitscale':
        X= scale(X)
    elif type == 'None':
        return X
    else:
        raise Exception('Invalid type for myScale')
    X[np.isnan(X)]=0
    X[np.isinf(X)]=0
    return X

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
    MSSHAPE = [numberOfSpectra,len(msdata_flat)/numberOfSpectra]
    msTrain = np.reshape(msdata_flat,MSSHAPE)

    #mx = msTrain.max(axis=1)
    #msTrain /= np.tile(mx[:,np.newaxis],(1,MSSHAPE[1]))


    #msTrainClean = normalize(msTrain)
    #trainFV = features(msTrainClean,featurelist=['autocorrelation'])
    #Scale features
    #trainFV_n = scale(trainFV)

    TESTFEATURES = ['fullSignal']
    TESTMODEL = 'svm_linear'
    MAXF=1000

    if retrain:
        #Load truth labels
        msTrainLabel = np.fromfile(spectraLabels,sep='\t')
        #1st prediction:
        msTrainClean = normalize(msTrain,learnNormalization=True)
        trainFV = features(msTrainClean,featurelist=TESTFEATURES)
        trainFV_n = myScale(trainFV,type='None')
        print trainFV_n.shape
        print msTrainLabel.shape

        #uscore = testModel(trainFV_n.T,Y=msTrainLabel,model='univariate')


        ############################
        df=pd.read_csv('output/varinfo_l1_l.txt',delimiter='\t')
        fToUse = np.array([int(s[1:]) for s in df['FeatureName'][:MAXF]])
        fToUse = np.append(fToUse,np.array([int(s[1:]) for s in df['FeatureName'][-MAXF:]]))
        ###########################

        trainModel = fitModel(trainFV_n[fToUse,:].T,msTrainLabel,model=TESTMODEL)
        trainPred=trainModel.predict(trainFV_n[fToUse,:].T)
        trainPred = binarizePred(trainPred)

        print 'Train Confusion matrix:'
        print confusion_matrix(msTrainLabel,trainPred)
        '''
        #extract top coefficients for second training:
        fToCorrelate= np.argsort(trainModel.coef_)[-20:]
        pickle.dump(fToCorrelate,open('fToCorrelate.pkl','wb'))
        trainFV = features(msTrainClean,featurelist=TESTFEATURES)
        trainFV_n = myScale(trainFV)

        #run a second prediction:
        trainModel = fitModel(trainFV_n.T,msTrainLabel,model=TESTMODEL).best_estimator_ #Lasso(alpha=1E-15).fit(trainFV_n.T,msTrainLabel)
        trainPred= trainModel.predict(trainFV_n.T)
        trainPred = binarizePred(trainPred)
        print confusion_matrix(msTrainLabel,trainPred)
        '''
        pickle.dump(trainModel,open(pickleName,'wb'))
    else:

        msTrainClean = normalize(msTrain,learnNormalization=False)
        #fToCorrelate=pickle.load(open('fToCorrelate.pkl','rb'))

        testFV = features(msTrainClean,featurelist=TESTFEATURES)
        testFV_n = myScale(testFV)
        trainModel = pickle.load(open(pickleName,'rb'))
        msTrainLabel = np.fromfile(spectraLabels,sep='\t')
        #testFV_n = testModel(testFV_n.T,model='pca').T
        ############################
        df=pd.read_csv('output/varinfo_l1_l.txt',delimiter='\t')
        fToUse = np.array([int(s[1:]) for s in df['FeatureName'][:MAXF]])
        fToUse = np.append(fToUse,np.array([int(s[1:]) for s in df['FeatureName'][-MAXF:]]))
        ###########################

        testPred = trainModel.predict(testFV_n[fToUse,:].T)
        testPred = binarizePred(testPred)
        print testPred
        print 'Test classification report'
        print classification_report(msTrainLabel,testPred)
        print 'Test Confusion Matrix:'
        print confusion_matrix(msTrainLabel,testPred)

