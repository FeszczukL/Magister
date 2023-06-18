import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn_lvq import GlvqModel, GmlvqModel, RslvqModel
from data_transform import data_extract,wavelet_transform
from sklearn_lvq.utils import _to_tango_colors, _tango_color
import signal_feature_extraction
from sklearn.model_selection import cross_val_score
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from learn import plot_learning_curve
###imported
def plot(data, target, target_p, prototype, prototype_label, p, title):
    p.scatter(data[:, 0], data[:, 1] ,c=_to_tango_colors(target, 0), alpha=0.5)
    
    p.scatter(data[:, 0], data[:, 1],c=_to_tango_colors(target_p, 0),marker='.')
    p.scatter(prototype[:, 0], prototype[:, 1],c=_tango_color('aluminium', 5), marker='D')
    try:
        p.scatter(prototype[:, 0], prototype[:, 1], s=60, c=_to_tango_colors(prototype_label, 0), marker='.')
    except:
        p.scatter(prototype[:, 0], prototype[:, 1], c=prototype_label, marker='.')
    p.axis('equal')
    # p.legend(prototype_label,fontsize=12)
    p.set_title(title, fontsize=22)
   
###
def GLVQ(x,y,cv_n):
    glvq=GlvqModel( gtol=1e-6,beta=5)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scores=cross_val_score(glvq,x,y,scoring="accuracy",cv=cv_n) # cross validation
    glvq.fit(X_train,y_train)
    scores_f=display_scores(scores,"Glvq")
    plot1 = plt.figure(figsize = (16, 9))
    ax = plt.axes()
    ax.set_title('GMLVQ')
    plot(glvq.project(X_train, 2),
          y_train, glvq.predict(X_train), glvq.project(glvq.w_, 2),
         glvq.c_w_, ax,"Classification scatter plot")
    plt.show()
    return scores_f,glvq

def GMLVQ(x,y,cv_n):
    gmlvq=GmlvqModel()
    gmlvq.fit(x,y)
    scores=cross_val_score(gmlvq,x,y,scoring="accuracy",cv=cv_n) # cross validation
    scores_f=display_scores(scores,"GMLVQ")
    # p4 = plt.subplot(121)
    # p4.set_title('GMLVQ')
    # plot(gmlvq.project(x, 2),
    #     y, gmlvq.predict(x), gmlvq.project(gmlvq.w_, 2),
    #     gmlvq.c_w_, p4)
    # plt.show()
    
    return scores_f

def RSLVQ(x,y,cv_n):
    rslvq=RslvqModel( gtol=1e-6)
    rslvq.fit(x,y)
    scores=cross_val_score(rslvq,x,y,scoring="accuracy",cv=cv_n) # cross validation
    scores_f=display_scores(scores,"Rslvq")
    return scores_f

# from the book
def display_scores(scores,name):
    print(f"Scores {name}: {scores}")
    print(f"Mean {name}: {scores.mean()}")
    print(f"Standard deviation {name}: {scores.std()}")
    return {'Scores':str(scores),'Mean':scores.mean(),'Standard deviation':scores.std()}

def feature_extract(data,window,step):
    object_list=[]
    feature_matrix=[]
    scaler = MinMaxScaler(feature_range=(0, 1))
    row=[]
    for idx,df in enumerate(data):
        # df=df.drop(columns=["time_s","label"],errors='ignore')
        features=signal_feature_extraction.signal_processing(df,window,step)
        scaled_features=scaler.fit_transform(features)
        scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
        row.append(scaled_features.iloc[0])
        df_concat=pd.concat([scaled_features[col] for col in scaled_features.columns])
        feature_matrix.append(df_concat.reset_index(drop=True))
    return np.array(feature_matrix),row

def plots_mode(x,y):
    fig, axes = plt.subplots(3, 3, figsize=(10, 15))

     ##################    Glvq     ##################
    title = r"GlvqModel Radius = 2"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    estimator = GlvqModel( gtol=1e-6,beta=2)
    plot_learning_curve(
        estimator, title, x, y, axes=axes[:,0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
        )
    
    ##################    Glvq     ##################
    title = r"GlvqModel Radius = 3"
    # SVC is more expensive so we do a lower number of CV iterations:
    estimator = GlvqModel( gtol=1e-6,beta=3)
    plot_learning_curve(
        estimator, title, x, y, axes=axes[:,1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
        )
    
    ##################    Glvq     ##################
    title = r"GlvqModel Radius = 4"
    # SVC is more expensive so we do a lower number of CV iterations:
    estimator = GlvqModel( gtol=1e-6,beta=4)
    plot_learning_curve(
        estimator, title, x, y, axes=axes[:,2], ylim=(0.7, 1.01), cv=cv, n_jobs=4
        )
    plt.show()

def feature_selection(x,y):
    from sklearn.ensemble import RandomForestRegressor
    from boruta import BorutaPy
    x=pd.DataFrame(x)
    x=x.reset_index(drop=True)
    # let's initialize a RF model 
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    # let's initialize Boruta
    feat_selector = BorutaPy(
        verbose=2,
        estimator=model,
        n_estimators='auto',
        max_iter=10  # number of iterations to perform
    )

    # train Boruta
    # N.B.: X and y must be numpy arrays
    feat_selector.fit(np.array(x), np.array(y))

    # print support and ranking for each feature
    print("\n------Support and Ranking for each feature------")
    for i in range(len(feat_selector.support_)):
        if feat_selector.support_[i]:
            print("Passes the test: ", x.columns[i],
                  " - Ranking: ", feat_selector.ranking_[i])
        else:
            print("Doesn't pass the test: ",
                  x.columns[i], " - Ranking: ", feat_selector.ranking_[i])
def main():
    #plotting
    #douczanie
    #wiecej cech DONE
    #testy falek DONE
    big_df=pd.DataFrame()
    scores_table=pd.DataFrame(columns=['Scores', 'Mean', 'Standard deviation'])
    window=1000
    step=1000
    li,labels=data_extract()
    pca = PCA(n_components=0.95)

    raw_data_features,row=feature_extract(li,window,step)
    # feature_selection(row,labels)
    print("Features extracted")

    # plots_mode(raw_data_features,labels)

    plots_mode(pca.fit_transform(raw_data_features),labels)

    scores_table=scores_table.append(GLVQ(pca.fit_transform(raw_data_features),labels,6),ignore_index=True)




    # wave_trans_list,labels=wavelet_transform(li) #returns list of 60 by 10000 elements
    # wavelet_features=feature_extract(wave_trans_list[0],window,step)
    # scores_table=scores_table.append(GLVQ(wavelet_features,labels,6),ignore_index=True)
    # scores_table=scores_table.append(GMLVQ(raw_data_features,labels,6),ignore_index=True)
    # scores_table=scores_table.append(RSLVQ(raw_data_features,labels,6),ignore_index=True)

    # for wave_list in wave_trans_list:
    #     wavelet_features=feature_extract(wave_list,window,step)
    #     scores_table=scores_table.append(GLVQ(wavelet_features,labels,6),ignore_index=True)

    # scores_table.index = ['Scores', 'Mean', 'Standard deviation']
    # scores_table[f"GMLVQ"]=GMLVQ(raw_data_features,labels,6)
    # scores_table[f"RSLVQ"]=RSLVQ(raw_data_features,labels,6)
    # scores_table[f"GMLVQ Wave"]=GMLVQ(wavelet_features,labels,6)
    # scores_table[f"RSLVQ Wave"]=RSLVQ(wavelet_features,labels,6)
    # scores_table.index = ['bior1.1','bior1.5','bior2.4','bior3.5','bior6.8','coif1','coif2','coif4',
    #            'coif14','coif15','db1','db12','db13','db14','db15','db16','db29','db34','db35'
    #            ,'db38','haar','rbio1.1','rbio2.2','rbio2.4','rbio2.8','rbio3.5','rbio3.9','rbio6.8','sym6','sym8','sym10','sym18','sym19']

    # scores_table.to_excel("Tabela.xlsx")

if __name__=="__main__":
    main()