#https://qiita.com/FukuharaYohei/items/c87f61aee2a24466d5d4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, RocCurveDisplay, ConfusionMatrixDisplay

train_csv = pd.read_csv('Titanic/train.csv')
test_csv = pd.read_csv('Titanic/test.csv')
print(train_csv.info())
print(test_csv.info())

DICT_SURVIVED = {0: '0: Dead', 1: '1: Survived'}

def arrange_bar(ax, sr):
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=30, horizontalalignment="center")
    ax.grid(axis='y', linestyle='dotted')
    [ax.text(i, count, count, horizontalalignment='center') for i, count in enumerate(sr)]

sr_survived = train_csv['Survived'].value_counts().rename(DICT_SURVIVED)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
sr_survived.plot.pie(autopct="%1.1f%%", ax=axes[0])
sr_survived.plot.bar(ax=axes[1])

arrange_bar(axes[1], sr_survived)

plt.show()