
from matplotlib.pyplot import *
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut,KFold
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.neural_network import BernoulliRBM

def printModel(trainModel):
    print '==============='
    print trainModel
    print trainModel.best_params_
    print trainModel.best_score_
    print trainModel.best_estimator_
    print '================'

def fitModel(X,Y,model='Lasso',scoring=None):

    model = model.lower()

    if model == 'lasso':
        params = {'alpha':np.logspace(-10,-8,10)}
        trainModel = GridSearchCV(Lasso(),params,scoring=scoring)
        trainModel.fit(X,Y,cv=LeaveOneOut(len(Y)))
        printModel(trainModel)
        return trainModel.best_estimator_
    elif model == 'lassocv':
        trainModel = LassoCV(cv=KFold(len(Y)/2,5)).fit(X,Y)
        return trainModel
    elif model == 'bayes':
        trainModel = GaussianNB().fit(X,Y)
        return trainModel
    elif model =='neural_net':
        params = {'n_components':[4,8,16,32,64,128],
                  'learning_rate':np.logspace(-3,0,4)}
        trainModel = BernoulliRBM(n_components=32,learning_rate=10E-3).fit(X,Y)
        return trainModel
    elif model == 'svm_linear':
        params={'C': np.logspace(-5,5,11)
            }
        trainModel = GridSearchCV(SVC(kernel='linear'),params,scoring=scoring)
        trainModel.fit(X,Y,cv=LeaveOneOut(len(Y)))
        printModel(trainModel)
        return trainModel.best_estimator_
    elif model == 'svm_rbf':
        params={'C': np.logspace(-5,5,11),
                'gamma': np.logspace(-4,0,5)
            }
        trainModel = GridSearchCV(SVC(kernel='rbf'),params,scoring=scoring)
        trainModel.fit(X,Y,cv=LeaveOneOut(len(Y)))
        printModel(trainModel)
        return trainModel.best_estimator_
    elif model =='logistic_regression':
        params = {'C':np.logspace(-2,2,9)}
        trainModel = GridSearchCV(LogisticRegression(),params,scoring=scoring)
        trainModel.fit(X,Y,cv=LeaveOneOut(len(Y))).best_estimator_
        printModel(trainModel)
        return trainModel.best_estimator_
    else:
        raise Exception('Invalid model input')



    return trainModel

def testModel(X,Y=None,model='univariate'):
    model = model.lower()
    if model=='pca':
        pca = PCA(n_components=50)
        return pca.fit_transform(X)
    if model == 'univariate':
        selector = SelectPercentile(f_classif,percentile=10).fit(X,Y)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        return scores


        #print X_pca.shape
        #marker={1:'*r',-1:'ob'}
        #figure()
        #for i,tl in enumerate(Y):
        #    plot(X_pca[i,0],X_pca[i,1],marker[tl],markersize=16)