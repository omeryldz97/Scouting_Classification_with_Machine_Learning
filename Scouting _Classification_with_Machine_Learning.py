import pandas as pd
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=Warning)
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",None)
pd.set_option("display.float_format", lambda x:"%.3f" %x)


#Note:Since the data set is confidential, it cannot be shared.
df=pd.read_csv("",sep=";")
df2=pd.read_csv("",sep=";")

df.shape
df2.shape


dff = pd.merge(df, df2, how='left', on=["task_response_id", "match_id", "evaluator_id", "player_id"])
df.columns


dff=dff[dff["position_id"]!=1]


pt=pd.pivot_table(dff,values="attribute_value",columns="attribute_id",index=["player_id","position_id","potential_label"])

pt=pt.reset_index(drop=False)
pt.columns=pt.columns.map(str)

le=LabelEncoder()
pt["potential_label"]=le.fit_transform(pt["potential_label"])
pt.head()

num_cols=pt.columns[3:]

scaler=StandardScaler()
pt[num_cols]=scaler.fit_transform(pt[num_cols])

y=pt["potential_label"]
X=pt.drop(["potential_label","player_id"],axis=1)

models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False)),
              ("LightGBM", LGBMClassifier())]

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score+" score:"+str(cvs))




def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("{0} Features".format(model))
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)
