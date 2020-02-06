import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
MODELS_DIR = 'models'

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

IGNORE_FEATURE_COLS = ['PassengerId', TARGET_VAR, KFOLD_VAR]

if __name__ == '__main__':
    df = pd.read_csv[TRAINING_DATA]
    df_test = pd.read_csv[TEST_DATA]
    train_df = df[df[KFOLD_VAR].isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df[KFOLD_VAR]==FOLD]

    ytrain = train_df[TARGET_VAR].values
    yvalid = valid_df[TARGET_VAR].values

    # drop useless features
    train_df = train_df.drop(IGNORE_FEATURE_COLS, axis=1)
    valid_df = valid_df.drop(IGNORE_FEATURE_COLS, axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(label_encoders, f"{MODELS_DIR}/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"{MODELS_DIR}/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"{MODELS_DIR}/{MODEL}_{FOLD}_columns.pkl")
