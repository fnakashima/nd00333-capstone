import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    #logger = logging.getLogger(__name__)
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.
def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'model.joblib'
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_filename)
    try:
        logger.info("Loading model from path:"+ model_path)
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.
input_sample = pd.DataFrame({"Gender": pd.Series([0], dtype="int64"), "Married": pd.Series([0], dtype="int64"), "Dependents": pd.Series([0], dtype="int64"), "Education": pd.Series([0], dtype="int64"), "Self_Employed": pd.Series([0], dtype="int64"), "ApplicantIncome": pd.Series([0], dtype="int64"), "CoapplicantIncome": pd.Series([0.0], dtype="float64"), "LoanAmount": pd.Series([0.0], dtype="float64"), "Loan_Amount_Term": pd.Series([0.0], dtype="float64"), "Credit_History": pd.Series([0.0], dtype="float64"), "Property_Area": pd.Series([0], dtype="int64")})
output_sample = np.array([0])

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
        