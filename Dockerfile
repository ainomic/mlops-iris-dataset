# Use the official Python image as the base image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Create the /app directory with appropriate permissions
RUN mkdir -p /app && chmod 777 /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variables for MLflow server
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT /app/mlflow_artifacts
ENV MLFLOW_SERVER_DEFAULT_FILE_STORE /app/mlflow_files

# Expose the MLflow server port
EXPOSE 5000

# Start the MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0"]
