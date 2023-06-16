import time

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import mlflow

# Set the tracking server to local server
mlflow.set_tracking_uri('http://localhost:5001')

# Set the logged model
logged_model = 'runs:/238469551abb4712be7f73e7bb21d00e/iris_rf_model'

# Set page title
st.title('IRIS category prediction app')


# Load data
@st.cache_data
def load_data():
    iris_data = load_iris()
    # Split the data into training and test sets. (0.8, 0.2) split.
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2, random_state=42)
    y_test = y_test.reshape(-1, 1)
    test = np.concatenate((X_test, y_test), axis=1)
    return test, iris_data.feature_names + ['target'], iris_data.target_names


test, feature_names, target_names = load_data()
data = pd.DataFrame(test, columns=feature_names)


# Show data
if st.sidebar.checkbox('Show raw test data'):
    st.subheader('Raw data')
    st.write(data)


# Input parameters
def user_input_features(index):
    # fetch data based on index from dataframe
    sepal_length = data.iloc[index, 0]
    sepal_width = data.iloc[index, 1]
    petal_length = data.iloc[index, 2]
    petal_width = data.iloc[index, 3]
    target = data.iloc[index, 4]
    return sepal_length, sepal_width, petal_length, petal_width, target


# index
index = st.sidebar.slider('index', 0, data.shape[0], 0)
# Set input parameters
sepal_length, sepal_width, petal_length, petal_width, target = user_input_features(
    index)

# Show input parameters
# Set subheader
st.subheader('User Input parameters')
col1, col2, col3, col4 = st.columns(4)
col1.write('sepal length')
col1.write(str(sepal_length))
col2.write('sepal width')
col2.write(str(sepal_width))
col3.write('petal length')
col3.write(str(petal_length))
col4.write('petal width')
col4.write(str(petal_width))


# Load model
@st.cache_data
def load_model():
    # Load model as a PyFuncModel.
    return mlflow.pyfunc.load_model(logged_model)


loaded_model = load_model()

# Predict
prediction = loaded_model.predict(pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]]))

col1, col2 = st.columns(2)
# Show actual target
col1.subheader('Actual target')
col1.write(f'{target_names[int(target)]} ({int(target)})')

# Show prediction
# Set subheader
col2.subheader('Prediction')
col2.write(f'{target_names[int(prediction)]} ({int(prediction)})')


def plot():
    X = test[:, :-1]
    actual = test[:, -1]
    predicted = loaded_model.predict(X)
    df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted,
        'Prediction': np.where(actual == predicted, 'Correct', 'Incorrect')
    })
    cols = plotly.colors.DEFAULT_PLOTLY_COLORS
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(x=df.index, y=df.Predicted,
                   marker=dict(color=cols[1]),
                   name="Predicted"),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df.Actual,
                   marker=dict(color=cols[0]),
                   name="Actual"),
        row=2, col=1)
    fig.add_vline(x=index, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(
        title_text="Actual vs Predicted with Shared X-Axes",
        xaxis_title="Index",
        yaxis_title="Class")
    st.plotly_chart(fig)


# Show plot
st.subheader('Actual vs Predicted')
plot()
