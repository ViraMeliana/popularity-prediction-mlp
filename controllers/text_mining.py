import jsonschema

from controllers.experimental.manual_perceptrons_v2 import ManualTrainTest

from rules.prediction_rule import *
from flask import Response, request, json
from resources.validations.error_messages import *


def index():
    response_text = '{ "message": "Hello, welcome to views predictions api" }'
    response = Response(response_text, 200, mimetype='application/json')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def find():
    request_data = request.get_json()
    try:
        testing = ManualTrainTest()
        jsonschema.validate(request_data, prediction_rules_schema)
        predict_result = testing.predict([request_data['title']])
        response = Response(json.dumps(str(predict_result)), 201, mimetype="application/json")
    except jsonschema.exceptions.ValidationError as exc:
        response = Response(error_message_helper(exc.message), 400, mimetype="application/json")
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

