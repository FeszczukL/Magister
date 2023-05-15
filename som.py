import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn_lvq import GlvqModel, GmlvqModel, RslvqModel
from data_transform import data_extract,wavelet_transform
from sklearn_lvq.utils import _to_tango_colors, _tango_color
import signal_feature_extraction
###imported
def plot(data, target, target_p, prototype, prototype_label, p):
    p.scatter(data[:, 0], data[:, 1], c=_to_tango_colors(target, 0), alpha=0.5)
    p.scatter(data[:, 0], data[:, 1], c=_to_tango_colors(target_p, 0),
              marker='.')
    p.scatter(prototype[:, 0], prototype[:, 1],
              c=_tango_color('aluminium', 5), marker='D')
    try:
        p.scatter(prototype[:, 0], prototype[:, 1], s=60,
                  c=_to_tango_colors(prototype_label, 0), marker='.')
    except:
        p.scatter(prototype[:, 0], prototype[:, 1], s=60,
                  c=_tango_color(prototype_label), marker='.')
    p.axis('equal')
###
def GLVQ(x,y):
    glvq=GlvqModel( gtol=1e-6,display=True)
    glvq.fit()

def GMLVQ(x,y):
    gmlvq=GmlvqModel()
    gmlvq.fit(x,y)
    p4 = plt.subplot(121)
    p4.set_title('GMLVQ')
    plot(gmlvq.project(x, 2),
        y, gmlvq.predict(x), gmlvq.project(gmlvq.w_, 2),
        gmlvq.c_w_, p4)
    plt.show()
    return
def RSLVQ(x,y):
    rslvq=RslvqModel( gtol=1e-6,display=True)
    rslvq.fit()

def main():
    li=data_extract()
    wave_trans_list,labels=wavelet_transform(li) #returns list of 60 by 10000 elements
    object_list=[]
    feature_matrix=[]
    for idx,df in enumerate(li):
        object_list.append(signal_feature_extraction.signal_processing(df.iloc[:,:8],1000,10))
        df_concat=pd.concat([object_list[idx][col] for col in object_list[idx].columns])
        feature_matrix.append(df_concat.reset_index(drop=True))
        
    
    GMLVQ(np.array(feature_matrix),labels)
if __name__=="__main__":
    main()