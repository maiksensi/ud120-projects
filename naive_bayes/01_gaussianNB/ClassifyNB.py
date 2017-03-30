def classify(features_train, labels_train):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    import prep_terrain_data
    train_x, train_y, _, _ = prep_terrain_data.makeTerrainData()
    clf.fit(train_x, train_y)
    return clf
