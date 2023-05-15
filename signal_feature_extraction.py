import scipy
import numpy as np
from scipy.integrate import quad
from sklearn.metrics import mean_squared_error
import statistics
import putemg_features
import pandas as pd
def signal_processing(input_signal,window,step):

    df_features= pd.DataFrame()
    
    df_features["Mav"]=[np.mean(np.abs(input_signal[column])) for column in input_signal.columns]
    # ssi=quad(input_signal,0,1) # Compute SSI
    df_features["Rms"]=[ np.sqrt(np.mean(np.square(input_signal[column]))) for column in input_signal.columns]
    
    df_features["Myog"] = [putemg_features.feature_myop(input_signal[column],window,step,0.5).values[0] for column in input_signal.columns]
    
    df_features["Var"] = [putemg_features.feature_var(input_signal[column],window,step).values[0] for column in input_signal.columns ]

    return df_features



