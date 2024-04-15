# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV, learning_curve
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, RobustScaler
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel, RFECV, f_classif, mutual_info_classif, f_regression, mutual_info_regression
# from sklearn.linear_model import SGDClassifier
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns

# def classify_df(df, features, target):
    
#     model = KNeighborsClassifier()
#     target_labels = df[target].unique()
#     codes = {label:code for code, label in enumerate(target_labels)}
#     y = df[target].map(codes)
#     X = df[features]
#     model.fit(X, y)
#     score = model.score(X, y)
#     return score 

# def plot_pca(df, target_label, features, n_components=2, ax=None):
#     keep_cols = [target_label] + features
#     df = df.loc[:,keep_cols]    
#     y = df.loc[:,[target_label]].values
#     x = df.loc[:, features].values

#     x = StandardScaler().fit_transform(x)

#     pca = PCA(n_components = n_components) # nb de dimensiosn sur lesquelles onr désire projeter nos données
#     # print(pca.explained_variance_)
#     # print(pca.explained_variance_ratio_)
#     principalComponents = pca.fit_transform(x)
#     principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#     finalDf = pd.concat([principalDf, df[[target_label]].reset_index()], axis = 1)

#     if ax is None:
#         fig, ax = plt.subplots()
#     ax.set_xlabel(f'PC1 : {np.round(pca.explained_variance_ratio_, 2)[0]}')
#     ax.set_ylabel(f'PC2 : {np.round(pca.explained_variance_ratio_, 2)[1]}')
#     ax.set_title(f'{n_components} component PCA')
#     targets = df[target_label].unique()

#     for target in targets:
#         indicesToKeep = finalDf[target_label] == target
#         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], s = 20)
#     ax.legend(targets)
#     # ax.grid()
#     return ax

# def plot_lda(df, target_label, features, n_components=2, ax=None):
#     keep_cols = [target_label] + features
#     df = df.loc[:,keep_cols]    
#     y = df.loc[:,[target_label]].values.ravel()
#     x = df.loc[:, features].values

#     x = StandardScaler().fit_transform(x)

#     lda = LinearDiscriminantAnalysis(n_components = n_components) # nb de dimensiosn sur lesquelles onr désire projeter nos données
#     principalComponents = lda.fit_transform(x, y)
#     principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#     finalDf = pd.concat([principalDf, df[[target_label]].reset_index()], axis = 1)

#     if ax is None:
#         fig, ax = plt.subplots()
#     # ax.set_xlabel('Principal Component 1', fontsize = 15)
#     # ax.set_ylabel('Principal Component 2', fontsize = 15)
#     ax.set_title('Linear Discriminant Analysis', fontsize = 20)
#     targets = df[target_label].unique()

#     for target in targets:
#         indicesToKeep = finalDf[target_label] == target
#         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], s = 20)
#     ax.legend(targets)
#     # ax.grid()
#     return ax


# def auto_classify_df(df, features, target, estimator_model=KNeighborsClassifier(), param_grid = {'n_neighbors':np.arange(1,80), 'metric':['euclidean','manhattan']}, verbose = False, show=False):
#     target_labels = df[target].unique()
#     codes = {label:code for code, label in enumerate(target_labels)}
#     y = df[target].map(codes)
#     X = df[features]

#     # train set & test set split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5) # découpe 80% train et 20% test

#     # grid search CV to search for best params in param grid
#     grid = GridSearchCV(estimator = estimator_model, param_grid=param_grid, cv=5)
#     grid.fit(X_train, y_train)

#     model = grid.best_estimator_

#     if verbose:
#         print(model)
#         print('best score :', grid.best_score_)
#         print(confusion_matrix(y_true=y_test, y_pred=model.predict(X_test)))

#     if show:
#         fig, axs = plt.subplots(ncols = 2, figsize = (15,7))

#         # validation curve to explore overfitting = overfitting si bon train > mauvais val
#         k = param_grid['n_neighbors'] # teste toutes les valeur pour l'hyperparamètre n_neighbors sur train set et val set de la cross validation
#         train_score, val_score = validation_curve(estimator=model, X=X_train, y=y_train, param_name='n_neighbors', param_range=k, cv = 5)

#         ax = axs[0]
#         ax.plot(k, val_score.mean(axis=1), label = 'validation')
#         ax.plot(k, train_score.mean(axis=1), label = 'train')
#         ax.set_ylabel('score')
#         ax.set_xlabel('n_neighbors')
#         ax.set_title('Validation Curve : Overfitting if train > validation')
#         ax.legend()
        
#         # learning curve to see if more data are useful
#         N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1,1,10), cv=5)
        
#         ax = axs[1]
#         ax.plot(N, train_score.mean(axis=1), label ='train')
#         ax.plot(N, val_score.mean(axis=1), label = 'validation')
#         ax.set_ylabel('score')
#         ax.set_xlabel('n points de data')
#         ax.set_title('Learning Curve : Do we need more data ?')
#         ax.legend()

#         plt.show()

#     return model.score(X_test, y_test)

# def auto_classify_df_preprocessed(df, features, target, show=True, n_selected_features = None):

#     # preprocessing
#     y = LabelBinarizer().fit_transform(df[target]) # encoding of strings in int

#     if n_selected_features is None:
#         X = df[features] 
#     else: # feature selection
#         selector = SelectKBest(f_classif, k = n_selected_features)
#         selector.fit_transform(df[features], df[target])
#         selected_features = np.array(features)[selector.get_support()]
#         rejected_features = np.array(features)[~selector.get_support()]
#         X = df.loc[:,selected_features]
        

#     # train set & test set split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5) # découpe 80% train et 20% test

#     # pipeline
#     model = Pipeline([('scaler', RobustScaler()), ('neigh', KNeighborsClassifier())])

#     # grid cv
#     param_grid = {'neigh__n_neighbors':np.arange(1,80), 'neigh__metric':['euclidean','manhattan']}
#     grid = GridSearchCV(model, param_grid=param_grid, cv=5)
#     grid.fit(X_train, y_train)

#     model = grid.best_estimator_

#     if show:
#         if not n_selected_features is None:
#             print('selected features :', selected_features)
#             print('rejected features :', rejected_features)
#         print(grid.best_estimator_)
#         print('best score :', grid.best_score_)
#         fig, axs = plt.subplots(ncols = 3, figsize = (25,7))

#         # confusion matrix
#         cm = confusion_matrix(y_test.argmax(axis = 1), model.predict(X_test).argmax(axis = 1))

#         ax = axs[0]
#         sns.heatmap(cm, annot=True, fmt='g', cbar = False, ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation
#         ax.set_xlabel('Predicted labels')
#         ax.set_ylabel('True labels')
#         ax.set_title('Confusion Matrix')
#         ax.xaxis.set_ticklabels(df[target].unique())
#         ax.yaxis.set_ticklabels(df[target].unique())

#         # validation curve to explore overfitting = overfitting si bon train > mauvais val
#         k = param_grid['neigh__n_neighbors'] # teste toutes les valeur pour l'hyperparamètre n_neighbors sur train set et val set de la cross validation
#         train_score, val_score = validation_curve(estimator=KNeighborsClassifier(), X=X_train, y=y_train, param_name='n_neighbors', param_range=k, cv = 5)

#         ax = axs[1]
#         ax.plot(k, val_score.mean(axis=1), label = 'validation')
#         ax.plot(k, train_score.mean(axis=1), label = 'train')
#         ax.set_ylabel('score')
#         ax.set_xlabel('n_neighbors')
#         ax.set_title('Validation Curve : Overfitting if train > validation')
#         ax.legend()
        
#         # learning curve to see if more data are useful
#         N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1,1,10), cv=5)
        
#         ax = axs[2]
#         ax.plot(N, train_score.mean(axis=1), label ='train')
#         ax.plot(N, val_score.mean(axis=1), label = 'validation')
#         ax.set_ylabel('score')
#         ax.set_xlabel('n points de data')
#         ax.set_title('Learning Curve : Do we need more data ?')
#         ax.legend()

#         plt.show()

#     return model.score(X_test, y_test)


# def raw_classify_df(df, features, target):
    
#     model = KNeighborsClassifier()
#     target_labels = df[target].unique()
#     # codes = {label:code for code, label in enumerate(target_labels)}
#     y = df[target]
#     X = df[features]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
#     model.fit(X_train, y_train)
#     score = model.score(X_test, y_test)
#     return score 