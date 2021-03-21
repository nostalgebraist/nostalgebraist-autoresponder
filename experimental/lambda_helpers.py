import json
import boto3

from bot_config import BotSpecificConstants

bot_specific_constants = BotSpecificConstants.load()
ml_lambda_function_name = bot_specific_constants.ml_lambda_function_name

lambda_client = boto3.client("lambda")


def request_ml_from_lambda(data: dict, n_concurrent: int = 1):
    resps = [
        lambda_client.invoke(
            FunctionName=ml_lambda_function_name,
            InvocationType="Event",
            Payload=json.dumps(data).encode("utf-8"),
        )
        for i in range(n_concurrent)
    ]
    return resps


def warm_lambda(n_concurrent: int = 5):
    data = {'id': '', 'hi': True}
    request_ml_from_lambda(data=data, n_concurrent=n_concurrent)


def parse_sns_request(request):
    data = json.loads(json.loads(request.get_data(as_text=True))['Message'])
    return data["responsePayload"]
