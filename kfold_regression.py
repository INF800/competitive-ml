import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection

# credits: Abishek Thakur
def create_folds(data, k):
    """data is df for regression dataset"""

    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=k)

    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins column and
    # return dataframe with folds
    data = data.drop("bins", axis=1)
    return data.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":

    df = pd.read_csv('inputs/train_clean.csv')
    df = create_folds(df, k=5)
    
    # print(df.skew())

    """
    save to csv
    """

    df["IDS"] = list(range(len(df)))
    df.to_csv("inputs/train_folds.csv", index=False)