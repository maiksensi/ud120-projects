def classify(features_train, labels_train):
    from sklearn import svm
    clf = svm.SVC(C=10000.0, kernel='rbf')
    import prep_terrain_data
    train_x, train_y, _, _ = prep_terrain_data.makeTerrainData()
    return clf.fit(train_x, train_y)
