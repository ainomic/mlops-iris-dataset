# MLOps on IRIS dataset

MLOps implementation using IRIS dataset as a practice dataset.

## Getting started

### Setup dev environment

1. Create virtual environment: `conda create -n mlops-iris-env -y python=3.8`
1. Activate virtual environment: `conda activate mlops-iris-env`
1. Install requirements: `pip install -r requirements.txt`
1. You're good to go!

## Explore the IRIS dataset

1. Activate virtual environment: `conda activate mlops-iris-env`
1. Change directory: `cd src`
1. Run [iris_dataset.py](src/iris_dataset.py): `python iris_dataset.py`
1. You should see the following output:

   ```text
    Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    Target names: ['setosa' 'versicolor' 'virginica']
    First 5 samples:
    [[5.1 3.5 1.4 0.2]
    [4.9 3.  1.4 0.2]
    [4.7 3.2 1.3 0.2]
    [4.6 3.1 1.5 0.2]
    [5.  3.6 1.4 0.2]]
    [0 0 0 0 0]
   ```

## Build and train the RandomForestClassifier model

1. Activate virtual environment: `conda activate mlops-iris-env`
1. Change directory: `cd src`
1. Run [train.py](src/train.py): `python train.py`
1. In another terminal with virtual environment activated, run `mlflow ui`. This will start the MLflow UI server at [localhost:5000](127.0.0.1:5000)
1. Browse to [localhost:5000](127.0.0.1:5000) and you should see the list of experiments with the parameters, model logged.

## Setup MLflow server

1. Build Docker container from `Dockerfile`: `docker build -t mlflow-server .`
1. Run Docker container: `docker run -d -p 5001:5000 mlflow-server`
1. Browse to [localhost:5001](localhost:5001) and you should see the list of experiments with the parameters, model logged.

## Run Flask API server

1. Activate virtual environment: `conda activate mlops-iris-env`
1. Change directory: `cd src`
1. Run [api.py](src/api.py): `python api.py`
1. This will run a local API server which is listening on [localhost:5002](localhost:5002)
1. You can test it by creating a POST request to __localhost:5002/predict__ endpoint with the following data:

   ```json
   {
      "data": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [6.7, 3.1, 5.6, 2.4]]
   }
   ```

   and, you should see the output as following:

   ```json
   [
      0,
      1,
      2
   ]
   ```

## Run Streamlit-based web app

1. Activate virtual environment: `conda activate mlops-iris-env`
1. Change directory: `cd src`
1. Run [app.py](src/app.py): `streamlit run app.py`
1. This should open a new browser window with the interactive web app at [localhost:8501](http://localhost:8501/).

## Blog series

1. [MLOps Intro: Build & Deploy RandomForest model with MLflow (Part-1)](https://ainomictech.medium.com/mlops-intro-build-deploy-randomforest-with-mlflow-part-1-49ba5308cf29)
1. [MLOps Intro: Build & Deploy RandomForest model with MLflow (Part-2)](https://ainomictech.medium.com/mlops-guide-build-deploy-randomforest-with-mlflow-part-2-d6baf0792fb1)
1. [MLOps Intro: Build & Deploy RandomForest model with MLflow (Part-3)](https://ainomictech.medium.com/mlops-intro-build-deploy-randomforest-with-mlflow-part-3-9c8de20a25a7)
1. [MLOps Intro: Build & Deploy RandomForest model with MLflow (Part-4)](https://ainomictech.medium.com/mlops-intro-build-deploy-randomforest-with-mlflow-part-4-b90978602f5b)
