# Databricks notebook source
!mlflow --version

# COMMAND ----------

def load_data(url):
    import pandas as pd
    # Load dataset
    data = pd.read_csv(filepath_or_buffer=url,sep=',')
    return data

def train_test_split(final_data,target_column):
    from sklearn.model_selection import train_test_split
    X = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify = y, random_state=47)
    return X_train, X_test, y_train, y_test

def training_basic_classifier(X_train,y_train):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train,y_train)
    
    return classifier


def predict_on_test_data(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def predict_prob_on_test_data(model,X_test):
    y_pred = model.predict_proba(X_test)
    return y_pred


def get_metrics(y_true, y_pred, y_pred_prob):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred,average='micro')
    recall = recall_score(y_true, y_pred,average='micro')
    entropy = log_loss(y_true, y_pred_prob)
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}


def create_roc_auc_plot(clf, X_data, y_data):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    metrics.plot_roc_curve(clf, X_data, y_data) 
    plt.savefig('/tmp/roc_auc_curve.png')
    
    
def create_confusion_matrix_plot(clf, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix
    plot_confusion_matrix(clf, X_test, y_test)
    plt.savefig('/tmp/confusion_matrix.png')
    
    


# COMMAND ----------

url = 'https://raw.githubusercontent.com/TripathiAshutosh/dataset/main/iris.csv'
data = load_data(url)
data.head()

# COMMAND ----------

target_column = 'class'
X_train, X_test, y_train, y_test = train_test_split(data, target_column)

# COMMAND ----------

X_test.head()

# COMMAND ----------

model = training_basic_classifier(X_train,y_train)

# COMMAND ----------

y_pred = predict_on_test_data(model,X_test)
print(y_pred)
y_pred_prob = predict_prob_on_test_data(model,X_test)
print(y_pred_prob)

# COMMAND ----------

run_metrics = get_metrics(y_test, y_pred, y_pred_prob)

# COMMAND ----------

run_metrics

# COMMAND ----------

create_confusion_matrix_plot(model, X_test, y_test)

# COMMAND ----------

def create_experiment(experiment_name,run_name, run_metrics,model, confusion_matrix_path = None, 
                      roc_auc_plot_path = None, run_params=None):
    import mlflow
    #mlflow.set_tracking_uri("http://localhost:5000") 
    #use above line if you want to use any database like sqlite as backend storage for model else comment this line
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        
        
        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_materix')
            
        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot")
        
        mlflow.set_tag("tag1", "Iris Classifier")
        mlflow.set_tags({"tag2":"Logistic Regression", "tag3":"Multiclassification using Ovr - One vs rest class"})
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("run_id", run_id)
    print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))

# COMMAND ----------

from datetime import datetime
experiment_name = "/Users/salman.kadaya@nagarro.com/iris_classifier_"+ str(datetime.now().strftime("%d-%m-%y")) ##basic classifier
run_name="iris_classifier_"+str(datetime.now().strftime("%d-%m-%y"))
name="iris-classifier"
create_experiment(experiment_name,run_name,run_metrics,model,'/tmp/confusion_matrix.png')

# COMMAND ----------

import mlflow
import mlflow.tracking

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
print(run_id)

model_uri = "runs:/{}/model".format(df.loc[0, "run_id"])

model_details = mlflow.register_model(model_uri , "iris-classifier")
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
print(latest_versions)
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

import mlflow
logged_model = "runs:/{}/model".format(df.loc[0, "run_id"])

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(X_test))

# COMMAND ----------

import mlflow.pyfunc

model_name = "iris-classifier"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

y_pred = model.predict(X_test)
print(y_pred)

sklearn_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
y_pred_prob = sklearn_model.predict_proba(X_test)
print(y_pred_prob)

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
registered_model = client.get_registered_model("iris-classifier")

# Get the latest version of the registered model
model_info = client.get_latest_versions(registered_model.name, stages=["Staging"])
print(model_info)
model_version = model_info[0].version
print(model_version)

client.transition_model_version_stage(
    name="iris-classifier",
    version=model_version ,
    stage="Production"
)

# COMMAND ----------

import mlflow.pyfunc

model_name = "iris-classifier"
stage = 'Production'

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)

y_pred = model.predict(X_test)
print(y_pred)

# COMMAND ----------

import mlflow.pyfunc

model_name = "iris-classifier"
stage = 'Production'

model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)

y_pred = model.predict([[6.7,3.3,5.7,2.1]])
print(y_pred)
y_pred_prob = model.predict_proba([[6.7,3.3,5.7,2.1]])
print(y_pred_prob)

# COMMAND ----------


