from src.preprocessing import(
    load_data,remove_duplicates,split_data,feature_scaling,apply_smote
)

df=load_data("data/raw/creditcard.csv")

df=remove_duplicates(df)

x_train, x_test, y_train, y_test = split_data(df)
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

#x_train_smote,y_train_smote= apply_smote(x_train,y_train)


#x_train_scaled, x_test_scaled,scaler = feature_scaling(x_train_smote, x_test)
x_train_scaled, x_test_scaled,scaler = feature_scaling(x_train, x_test)

from src.model import get_model
#model = get_model("rf")
#model = get_model("xgb",scale_pos_weight=scale_pos_weight)
model = get_model("stacking",
                  scale_pos_weight=scale_pos_weight)

from src.train import train_model
trained_model = train_model(
    model,
    #x_train_scaled, y_train_smote)
    x_train_scaled, y_train)

from src.save_model import save_artifacts
save_artifacts(
    trained_model,
    scaler
)

from src.evaluate import evaluate
metrics= evaluate(
   trained_model,
    x_test_scaled,
    y_test,




)