# Databricks notebook source
# Configure the storage account
#spark.conf.set("fs.azure.account.key.mbusasa.blob.core.windows.net", "<KEY>")

# Read data from container
#final_data = spark.read.format("csv").option("header", "true").load("wasbs://abd-mlops@mbusasa.blob.core.windows.net/datafile/diabetes.csv")
#display(final_data)

# COMMAND ----------

import pandas as pd
# Load dataset
data=pd.read_csv("/dbfs/FileStore/shared_uploads/salman.kadaya@nagarro.com/diabetes.csv")
data.head()

# Drop null values
final_data = data.dropna()

target_column = 'Outcome'
from sklearn.model_selection import train_test_split
X = final_data.loc[:, final_data.columns != target_column]
y = final_data.loc[:, final_data.columns == target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify = y, random_state=47)
print(X_train)
print(y_train)


# COMMAND ----------

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# COMMAND ----------

import pandas as pd

# Load dataset
diabetes_data=pd.read_csv("/dbfs/FileStore/shared_uploads/salman.kadaya@nagarro.com/diabetes.csv")
diabetes_data.head()

# Drop null values
diabetes_data = diabetes_data.dropna()

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(diabetes_data)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, diabetes_data["Outcome"], test_size=0.2)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model
lr = LogisticRegression()

# Train the model on the training data
lr.fit(X_train, y_train)


# COMMAND ----------

y_pred = lr.predict(X_test)
print(y_pred)
y_pred_prob = lr.predict_proba(X_test)
print(y_pred_prob)

# COMMAND ----------

from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred,average='micro')
recall = recall_score(y_test, y_pred,average='micro')
entropy = log_loss(y_test, y_pred_prob)
run_metrics = {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}
run_metrics

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn import metrics
metrics.plot_roc_curve(lr, X_train, y_train) 
plt.savefig('/tmp/roc_auc_curve.png')

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(lr, X_test, y_test)
plt.savefig('/tmp/confusion_matrix.png')

# COMMAND ----------

import mlflow
mlflow.set_experiment("/Users/salman.kadaya@nagarro.com/diabetes_experiment")


# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

experiment_name="/Users/salman.kadaya@nagarro.com/diabetes"
run_name="diabetes_run"
confusion_matrix_path="/tmp/confusion_matrix.png"
roc_auc_plot_path="/tmp/roc_auc_curve.png"
model="lr"
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=run_name) as run:
    for metric in run_metrics:
        mlflow.log_metric(metric, run_metrics[metric])
    mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')
    mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
    mlflow.set_tag("tag1", "Diabetes Finder")
    mlflow.set_tags({"tag2":"Logistic Regression", "tag3":"Multiclassification using Ovr - One vs rest class"})
    signature = infer_signature(diabetes_data, lr.predict_proba(diabetes_data))
    name = "model-sk"
    mlflow.sklearn.log_model(model, name, signature=signature)
    #mlflow.sklearn.log_model(model, "model",registered_model_name="diabetes-finder")
    run_id = mlflow.active_run().info.run_id
    mlflow.set_tag("run_id", run_id)
mlflow.end_run()

print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))

# COMMAND ----------

# Get Experiment Information
experiment_info = mlflow.get_experiment_by_name(name=experiment_name)
experiment_id = experiment_info.experiment_id
print(experiment_id)

# Get Metadata Information
df = mlflow.search_runs([experiment_id])
df = df.sort_values(by=["end_time"], ascending=False)
run_id = df.loc[0, "run_id"]
artifact_uri = df.loc[0, "artifact_uri"]
print(artifact_uri)

# Register Model
model_details = mlflow.register_model(
    model_uri=artifact_uri + "/{0}".format(name), name=name
)
model_version = model_details.version

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
# Transition to Staging
client = MlflowClient()
stage = "Staging"
cur_date = datetime.today().date()
latest_versions = client.get_latest_versions(name=name)
description = (
    f"The model version {model_version} was transitioned to {stage} on {cur_date}"
)

client.transition_model_version_stage(
    name=name, version=model_version, stage=stage, archive_existing_versions=True
)

client.update_model_version(
    name=name, version=model_version, description=description
)

# COMMAND ----------

import pickle
file_to_store = {
    "name": name,
    "experiment_id": experiment_id,
    "run_id": run_id,
    "artifact_uri": artifact_uri,
    "model_version": model_version,
}

path_to_store = "/dbfs/FileStore/shared_uploads/salman.kadaya@nagarro.com/"

with open(path_to_store + "mlflow_info.json", "wb") as f:
    pickle.dump(file_to_store, f)

with open(path_to_store + "mlflow_info.json", "rb") as f:
    file_to_store = pickle.load(f)

print(file_to_store)

# COMMAND ----------

model = mlflow.sklearn.load_model(
    model_uri=f"models:/{name}/{model_version}"
)

y_pred = lr.predict(X_test)
print(y_pred)

# COMMAND ----------


