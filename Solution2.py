# -*- coding: utf-8 -*-
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/dataTrain.csv')
test_data = pd.read_csv('data/dataA.csv')
submission = pd.read_csv('data/submit_example_A.csv')
data_nolabel = pd.read_csv('data/dataNoLabel.csv')
print(f'train_data.shape = {train_data.shape}')
print(f'test_data.shape  = {test_data.shape}')

train_data['f47'] = train_data['f1'] * 10 + train_data['f2']
test_data['f47'] = test_data['f1'] * 10 + test_data['f2']

loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]] + 1)

# 暴力 Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]] + 1)

cat_columns = ['f3']
data = pd.concat([train_data, test_data])

for col in cat_columns:
    lb = LabelEncoder()
    lb.fit(data[col])
    train_data[col] = lb.transform(train_data[col])
    test_data[col] = lb.transform(test_data[col])

num_columns = [col for col in train_data.columns if col not in ['id', 'label', 'f3']]
feature_columns = num_columns + cat_columns
target = 'label'

train = train_data[feature_columns]
label = train_data[target]
test = test_data[feature_columns]


def model_train(model, model_name, kfold=5):
    oof_preds = np.zeros((train.shape[0]))
    test_preds = np.zeros(test.shape[0])
    skf = StratifiedKFold(n_splits=kfold)
    print(f"Model = {model_name}")
    for k, (train_index, test_index) in enumerate(skf.split(train, label)):
        x_train, x_test = train.iloc[train_index, :], train.iloc[test_index, :]
        y_train, y_test = label.iloc[train_index], label.iloc[test_index]

        model.fit(x_train, y_train)

        y_pred = model.predict_proba(x_test)[:, 1]
        oof_preds[test_index] = y_pred.ravel()
        auc = roc_auc_score(y_test, y_pred)
        print("- KFold = %d, val_auc = %.4f" % (k, auc))
        test_fold_preds = model.predict_proba(test)[:, 1]
        test_preds += test_fold_preds.ravel()
    print("Overall Model = %s, AUC = %.4f" % (model_name, roc_auc_score(label, oof_preds)))
    return test_preds / kfold


gbc = GradientBoostingClassifier()
gbc_test_preds = model_train(gbc, "GradientBoostingClassifier", 60)

train = train[:50000]
label = label[:50000]

gbc = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.2,
    max_depth=5
)
hgbc = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=5
)
xgbc = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.2
)
gbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    num_leaves=2 ** 6,
    max_depth=8,
    colsample_bytree=0.8,
    subsample_freq=1,
    max_bin=255,
    learning_rate=0.04,
    n_estimators=100,
    metrics='auc'
)
cbc = CatBoostClassifier(
    iterations=210,
    depth=6,
    learning_rate=0.04,
    l2_leaf_reg=1,
    loss_function='Logloss',
    verbose=0
)
estimators = [
    ('gbc', gbc),
    ('xgbc', xgbc),
    ('hgbc', hgbc),
    ('gbm', gbm),
    ('cbc', cbc)
]
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
X_train, X_test, y_train, y_test = train_test_split(
    train, label, stratify=label, random_state=2022)

clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)

ff = []
for col in feature_columns:
    x_test = X_test.copy()
    x_test[col] = 0
    auc1 = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    if auc1 < auc:
        ff.append(col)
    print('%5s | %.8f | %.8f' % (col, auc1, auc1 - auc))

clf.fit(X_train[ff], y_train)
y_pred = clf.predict_proba(X_test[ff])[:, 1]
auc = roc_auc_score(y_test, y_pred)
print('auc = %.8f' % auc)

train = train[ff]
test = test[ff]

clf_test_preds = model_train(clf, "StackingClassifier", 10)

submission['label'] = clf_test_preds
submission.to_csv('submission.csv', index=False)
