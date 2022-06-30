def classify_df(df, features, target):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    target_labels = df[target].unique()
    codes = {label:code for code, label in enumerate(target_labels)}
    y = df[target].map(codes)
    X = df[features]
    model.fit(X, y)
    score = model.score(X, y)
    return score 
