from imblearn.over_sampling import RandomOverSampler

def balanceDatasetWithRandomOversampling(X, y):
    ros = RandomOverSampler(random_state = 55)
    X2, y2 = ros.fit_resample(X, y)
    return X2, y2