
import numpy as np

from msDetect import *

if __name__=='__main__':
    spectraFile = open('../sample_data/cancer_train_100.data','rb')
    numberOfSpectra = 10000

    msdata_flat=np.fromfile(spectraFile,sep='\t')
    MSSHAPE = [numberOfSpectra,len(msdata_flat)/numberOfSpectra]
    msTrain = np.reshape(msdata_flat,MSSHAPE)

    truth = np.fromfile(open('../sample_data/cancer_train_100.labels','rb'),sep='\t')
    truth[truth==-1]=0

    msTrain=normalize(msTrain,True)

    try:
        vwFile = open('../sample_data/vw_train_n_l','wb')
        for j,col in enumerate(range(MSSHAPE[1])):
            row = str(int(truth[col]))+' \''+str(j)+' |'
            for i,fv in enumerate(msTrain[:,col]):
                row+=(' '+str(i)+':'+str(fv))
            #print row

            vwFile.write(row)
            vwFile.write('\n')
    except:
        print 'exception'
    finally:
        vwFile.close()