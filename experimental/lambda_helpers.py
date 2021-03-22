import json
import boto3
import uuid
import concurrent.futures as cf
from datetime import datetime
from functools import partial

from bot_config import BotSpecificConstants
from bridge_shared import wait_for_result, bridge_service_unique_id

bot_specific_constants = BotSpecificConstants.load()
ml_lambda_function_name = bot_specific_constants.ml_lambda_function_name

WARM_MAX_HEALTHCHECK_SECONDS = 5
ASSUME_WARM_WITHIN_SECONDS = 60
RECHECK_SECONDS = 1

wait_for_lambda_result = partial(wait_for_result,
                                 wait_first_time=RECHECK_SECONDS,
                                 wait_recheck_time=RECHECK_SECONDS,
                                 verbose=False,
                                 return_turnaround_time=True,
                                 )

lambda_client = boto3.client("lambda")


def _send_one_lambda_request(data: dict):
    return lambda_client.invoke(
        FunctionName=ml_lambda_function_name,
        InvocationType="Event",
        Payload=json.dumps(data).encode("utf-8"),
    )


def request_ml_from_lambda(data: dict, n_concurrent: int = 1):
    resps = [
        _send_one_lambda_request(data)
        for i in range(n_concurrent)
    ]
    return resps


def parse_sns_request(request):
    data = json.loads(json.loads(request.get_data(as_text=True))['Message'])
    return data["responsePayload"]


class TrackedLambda:
    def __init__(self, lambda_uid: str, last_response_time: datetime):
        self.lambda_uid = lambda_uid
        self.last_response_time = last_response_time

    @property
    def trusted_as_warm(self) -> bool:
        delta = datetime.now() - self.last_response_time
        secs = delta.total_seconds()
        return secs < ASSUME_WARM_WITHIN_SECONDS


def secure_n_warm_lambdas(n: int = 1):
    bridge_ids = [str(uuid.uuid4()) for i in range(n)]

    for bridge_id in bridge_ids:
        data = {'hi': True, 'id': bridge_id}
        _send_one_lambda_request(data=data)

    futures = []
    executor = cf.ProcessPoolExecutor(max_workers=n)
    for bridge_id in bridge_ids:
        futures.append(executor.submit(wait_for_lambda_result, new_id=bridge_id))

    lambdas = []

    for future in cf.as_completed(futures):
        result, time_sec = future.result()
        if 'lambda_uid' in result:
            t = datetime.now()
            lambdas.append(TrackedLambda(lambda_uid=result['lambda_uid'], last_response_time=t))
        else:
            print(f"weirdness: didn't find lambda_uid, have data {result}")

    executor.shutdown()

    return lambdas


class LambdaPool:
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.lambdas = []
        self.lambdas_occupied = 0

    def initialize(self):
        self.lambdas = secure_n_warm_lambdas(n=self.n_workers)

    def _request(self, data: dict):
        pass
