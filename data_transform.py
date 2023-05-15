import pandas as pd
from pathlib import Path
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pywt

this_folder = (Path(__file__).parent / "..").resolve()
# Returns list of data frames, each data frame is one data point
def data_extract() -> list:
    path_list = {
        "path_2": Path(f"{this_folder}\\Magisterka\\data\\2"),
        "path_3": Path(f"{this_folder}\\Magisterka\\data\\3"),
        "path_4": Path(f"{this_folder}\\Magisterka\\data\\4"),
        "path_5": Path(f"{this_folder}\\Magisterka\\data\\5"),
        "path_6": Path(f"{this_folder}\\Magisterka\\data\\6"),
        "path_7": Path(f"{this_folder}\\Magisterka\\data\\7"),
    }
    li = []
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
            df=df.drop([1, 3,5,7,9,11,13,14], axis=1)  
            df=df.replace(',', '.',regex=True)
            df = df.astype(float)
            df.columns=[0,1,2,3,4,5,6,7]
            df['time_s']=df.index*0.001 + 0.001
            df['label']=idx
            li.append(df)
    return li

def wavelet_transform(li:list) -> list:

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import FunctionTransformer
    import pywt

    wave_trans_list=[]
    wavelet = 'sym2'

    def wtf(X):
        coeffs = pywt.dwt(X, wavelet)
        # pywt.plotting.plot_dwt_tree(coeffs, ax=ax)
        return np.concatenate(coeffs,axis=1)

    # create the pipeline object that includes wavelet transform and scaling steps
    pipeline = Pipeline([
        ('wavelet_transform', FunctionTransformer(wtf)),
        # ('feature_selection', SelectKBest(score_func=f_regression, k=10)),
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ])
    labels=[]
    # # transform the test data using the fitted pipeline
    # apply the pipeline to the data
    for idx,X in enumerate(li):
        X=X.iloc[:,:8]
        y=li[0]['time_s']
        labels.append(li[idx]['label'][0])
        X_transformed = pipeline.fit_transform(X)
        wave_trans_list.append(X_transformed)
    wave_concat=[]
    # view the transformed data
    # X_transformed=pd.DataFrame(X_transformed)
    # X_transformed['time_s']=X_transformed.index*0.001 + 0.001
    # ploting_channels(X_transformed,1)
    for sample in wave_trans_list:
        wave_concat.append(np.concatenate(sample))
    return wave_concat,labels

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
    
    