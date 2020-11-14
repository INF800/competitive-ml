import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


PATH_TO_TRAIN_FOLDS_CSV = "inputs/train_folds.csv"



def run_training(fold):
    
    df = pd.read_csv(PATH_TO_TRAIN_FOLDS_CSV)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Xs, ys
    ytrain = df_train['target']
    xtrain = df_train.drop(['target', 'kfold', 'IDS'], axis=1)
    yvalid = df_valid['target']
    xvalid = df_valid.drop(['target', 'kfold', 'IDS'], axis=1)

    # scaling
    scaler.fit(xtrain)
    scaler.fit(xvalid)
    scaler.transform(xtrain)
    scaler.transform(xvalid)

    # train and predict
    model = LinearRegression().fit(xtrain, ytrain)
    ypred = model.predict(xvalid)

    # calc rmse
    mse = metrics.mean_squared_error(yvalid, ypred)
    print(f"{fold}\t: {np.sqrt(mse)}")

    # add to df_valid
    df_valid['linreg_pred'] = ypred

    return df_valid


if __name__ == '__main__':
    for k in range(5):
        run_training(k)
