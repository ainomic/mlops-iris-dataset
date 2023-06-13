import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the IRIS dataset
data = load_iris()

# Split the data into training and test sets. (0.8, 0.2) split.
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# Build and Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)  # Model accuracy: 1.0

# Log model hyperparameters and metrics to the MLflow server
with mlflow.start_run():
    # Log the model
    mlflow.sklearn.log_model(model, "iris_rf_model")
    # Log the parameters
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("random_state", model.random_state)
    # Log the accuracy score
    mlflow.log_metric("accuracy", accuracy)

    # Save the MLflow run ID
    run_id = mlflow.active_run().info.run_id
    print("MLflow Run ID:", run_id)
