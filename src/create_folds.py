import pandas as pd
from sklearn import model_selection

TRAIN_PATH = 'data/train.csv'
TARGET_VAR = 'Survived'
KFOLD_VAR  = 'kfold'
TRAINFOLDS_PATH = 'data/train_folds.csv'
N_SPLITS = 5

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_PATH)
    df[KFOLD_VAR] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=N_SPLITS, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[TARGET_VAR].values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, KFOLD_VAR] = fold

    df.to_csv(TRAINFOLDS_PATH, index=False)
