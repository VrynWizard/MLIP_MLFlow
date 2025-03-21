import mlflow
import numpy as np
import pandas as pd

# TODO: Set tht MLFlow server uri
uri = "http://127.0.0.1:6001"
mlflow.set_tracking_uri(uri=uri)

# TODO: Provide model path/url
logged_model = 'mlflow-artifacts:/231042642541627973/b55911b0af3340a7b3f8c1cc37faea25/artifacts/'

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)


# Input a random datapoint
np.random.seed(42)
data = np.random.rand(1, 64)

# TODO: Predict the output for the data. You might need to use a pandas DataFrame due to a constraint from MLFlow.
prediction = loaded_model.predict(pd.DataFrame(data))

# Print out prediction result
print(prediction)
