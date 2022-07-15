from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def classify_df(df, features, target):
    
    model = KNeighborsClassifier()
    target_labels = df[target].unique()
    codes = {label:code for code, label in enumerate(target_labels)}
    y = df[target].map(codes)
    X = df[features]
    model.fit(X, y)
    score = model.score(X, y)
    return score 

def plot_pca(df, target_label, features, n_components=2, ax=None):
    keep_cols = [target_label] + features
    df = df.loc[:,keep_cols]    
    y = df.loc[:,[target_label]].values
    x = df.loc[:, features].values

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components = n_components) # nb de dimensiosn sur lesquelles onr désire projeter nos données
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[[target_label]]], axis = 1)

    if ax is None:
        fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = df[target_label].unique()

    for target in targets:
        indicesToKeep = finalDf[target_label] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , s = 50)
    ax.legend(targets)
    ax.grid()
    return ax