import joblib
import pandas as pd
import seaborn as sns
import os
from sklearn.datasets import fetch_openml

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression

df = fetch_openml(name="titanic", version=1, as_frame=True).frame

df=df[["pclass","sex","age","fare","survived"]]
df["age"]=df["age"].fillna(df["age"].median())
x=df.drop("survived",axis=1)
y=df["survived"]
numeric_features=["age","fare"]
categorical_features=["sex","pclass"]

numeric_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scalar",StandardScaler())
])

#------------------
# CATEGORICAL PIPELINE
#---------------------

categorical_pipeline=Pipeline([
     ("imputer",SimpleImputer(strategy="most_frequent")),
     ("encoder",OrdinalEncoder())
])

#----------------------
# COMBINE
#----------------------

preprocessor=ColumnTransformer([
    ("num",numeric_pipeline,numeric_features),
    ("cat",categorical_pipeline,categorical_features)
])

#--------------------
# FINAL PIPELINE
#--------------------

pipeline=Pipeline([
    ("preprocessor",preprocessor),
    ("classifier",LogisticRegression())
])

pipeline.fit(x,y)

#Create the model directory if it doesn't exist
os.makedirs("model",exist_ok=True)

joblib.dump(pipeline,"model/pipeline.pkl")
print("pipeline saved successfully")
