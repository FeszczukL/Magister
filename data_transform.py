import pandas as pd
from pathlib import Path
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pywt

this_folder = (Path(__file__).parent / "..").resolve()
# Returns list of data frames, each data frame is one data point
def data_extract() -> list:
    # path_list = {
    #     "path_2": Path(f"{this_folder}\\Magisterka\\data\\2"),
    #     "path_3": Path(f"{this_folder}\\Magisterka\\data\\3"),
    #     "path_4": Path(f"{this_folder}\\Magisterka\\data\\4"),
    #     "path_5": Path(f"{this_folder}\\Magisterka\\data\\5"),
    #     "path_6": Path(f"{this_folder}\\Magisterka\\data\\6"),
    #     "path_7": Path(f"{this_folder}\\Magisterka\\data\\7"),
    # }
    path_list = {
        "path_2": Path(f"{this_folder}\\Magisterka\\data\\dataset\\2"),
        "path_3": Path(f"{this_folder}\\Magisterka\\data\\dataset\\3"),
        "path_4": Path(f"{this_folder}\\Magisterka\\data\\dataset\\4"),
        "path_5": Path(f"{this_folder}\\Magisterka\\data\\dataset\\5"),
        "path_6": Path(f"{this_folder}\\Magisterka\\data\\dataset\\6"),
        "path_1": Path(f"{this_folder}\\Magisterka\\data\\dataset\\1"),
    }
    li = []
    label=[]
    for idx,key in enumerate(path_list):
        if not (os.path.exists(path_list[key])):
            print(["Could not find ", path_list[key]])
        file_list = [
            log
            for log in path_list[key].glob("*.csv")
        ]

        for filename in file_list:
            #Load data, drop unnecessary columns, replace commas, turn string to float
            df = pd.read_csv(filename, index_col=None, header=None,sep=";")
            # df=df.drop([1, 3,5,7,9,11,13,14], axis=1)  
            df=df.replace(',', '.',regex=True)
            df = df.astype(float)
            # df.columns=[0,1,2,3,4,5,6,7]
            df['time_s']=df.index*0.001 + 0.001
            df['label']=idx
            label.append(idx)
            li.append(df)
    return li,label

def wavelet_transform(li:list) -> list:

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import FunctionTransformer
    import pywt


    # wavelet = ['bior1.1','bior1.5','bior2.4','bior3.5','bior6.8','coif1','coif2','coif4',
    #            'coif14','coif15','db1','db12','db13','db14','db15','db16','db29','db34','db35'
    #            ,'db38','haar','rbio1.1','rbio2.2','rbio2.4','rbio2.8','rbio3.5','rbio3.9','rbio6.8','sym6','sym8','sym10','sym18','sym19']
    wavelet=['bior1.1','bior1.5']

    # cA2, cD2, cD1 = pywt.wavedec(x, db1, mode='constant', level=2)
    def wtf(X):
        # coeffs = pywt.dwt(X, wavelet,mode='symmetric')
        n=pywt.dwt_max_level(len(X), wavelet_obj)
        coeffs = pywt.wavedec(X, wavelet_obj,mode='symmetric',level=n-1)
        # pywt.plotting.plot_dwt_tree(coeffs, ax=ax)
        return np.concatenate(coeffs,axis=1)
        # return coeffs[0]

    # create the pipeline object that includes wavelet transform and scaling steps
    pipeline = Pipeline([
        # ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('wavelet_transform', FunctionTransformer(wtf)),
        ('scaler2', MinMaxScaler(feature_range=(0, 1)))
        # ('feature_selection', SelectKBest(score_func=f_regression, k=10)),
        
    ])
    labels=[]
    wave_test=[]
    flag=True
    # apply the pipeline to the data
    for wavelet_obj in wavelet:
        wave_trans_list=[]
        
        for idx,X in enumerate(li):
            if flag:
                labels.append(X["label"][0])
            X=X.iloc[:,:8]
            X_transformed = pipeline.fit_transform(X,wavelet_obj)
            wave_trans_list.append(pd.DataFrame(X_transformed))
        flag=False
        wave_test.append(wave_trans_list)


 
    return wave_test,labels

def fourier_trans(signal,gamma,IP_v,IPS_v,e):
    G_prim=[]
    s=[]
    selected_f=[]
    #e - wanted number of features

    new_s=np.fft.rfft(signal) 

    def sigma_sum(start, end, expression):
        return sum(expression(i) for i in range(start, end))
    def harmonic_average(i):
        return (G_prim[i]/IPE_v-IPS_v+1)
    
    #Averaging of the spectral density
    # G′(n)= G′(n −1) + γ (G(n) − G′(n −1)) 
    for idx,i in enumerate(new_s):
        if idx==0:
            G_prim.append(i)
        else:
            G_prim.append(G_prim[idx-1]+gamma*(i-G_prim[idx-1]))

    #smoothing with averaging the harmonics with their neighbor
    for idx,i in enumerate(G_prim):
        if (idx-IP_v) < 0:
            IPS_v=0
        else:
            IPS_v=idx-IP_v
        if idx+IP_v>len(G_prim):
            IPE_v=len(G_prim)
        else:
            IPE_v=idx+IP_v
        s.append(sigma_sum(IPS_v, IPE_v, harmonic_average))

    s=[i.real for i in s]

    #linear selection
    for k in range(0,e):
        selected_f.append(s[int(k*(len(s)-1)/(e-1))])

    return selected_f

if __name__=="__main__":
    li=data_extract()
    
    