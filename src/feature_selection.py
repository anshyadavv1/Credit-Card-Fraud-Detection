from sklearn.feature_selection import SelectFromModel
def select_feature(model,x_train,y_train,x_test):
    selector = SelectFromModel(estimator=model, prefit=True,threshold="mean")
    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)
    return x_train_selected,x_test_selected,selector