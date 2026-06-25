import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df=pd.read_csv(path)
    df.head()
    return df

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def split_data(df):
    x = df.drop("Class", axis=1)
    y = df["Class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test

def feature_scaling(x_train,x_test):
    scaler = StandardScaler()

    X_train_scaled = x_train.copy()

    X_test_scaled = x_test.copy()

    X_train_scaled[["Amount", "Time"]] = scaler.fit_transform(

        x_train[["Amount", "Time"]]

    )

    X_test_scaled[["Amount", "Time"]] = scaler.transform(

        x_test[["Amount", "Time"]])
    X_train_scaled[["Amount", "Time"]].describe()
    return X_train_scaled, X_test_scaled, scaler