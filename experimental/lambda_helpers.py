import json
import boto3
import uuid
import time
import concurrent.futures as cf
from datetime import datetime
from functools import partial
from collections import defaultdict

from bot_config import BotSpecificConstants
from bridge_shared import wait_for_result

bot_specific_constants = BotSpecificConstants.load()
ml_lambda_function_name = bot_specific_constants.ml_lambda_function_name

WARM_MAX_HEALTHCHECK_SECONDS = 5
ASSUME_WARM_WITHIN_SECONDS = 60
ASSUME_COLD_WITHIN_SECONDS = 60 * 10
RECHECK_SECONDS = 5
STAGGER_LAMBDAS_WAIT_SEC = 5
STAGGER_LAMBDA_REQUESTS_SEC = 0.33

wait_for_lambda_result = partial(
    wait_for_result,
    wait_first_time=RECHECK_SECONDS,
    wait_recheck_time=RECHECK_SECONDS,
    verbose=False,
    return_times=True,
)

lambda_client = boto3.client("lambda")


def _send_one_lambda_request(data: dict):
    time.sleep(STAGGER_LAMBDA_REQUESTS_SEC)
    data['request_id'] = str(uuid.uuid4())
    lambda_client.invoke(
        FunctionName=ml_lambda_function_name,
        InvocationType="Event",
        Payload=json.dumps(data).encode("utf-8"),
    )
    return data['request_id']


def request_ml_from_lambda(data: dict, n_concurrent: int = 1):
    request_ids = [_send_one_lambda_request(data) for i in range(n_concurrent)]
    return request_ids


def parse_sns_request(request):
    data = json.loads(json.loads(request.get_data(as_text=True))["Message"])
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
    request_ids = []

    for _ in range(n):
        data = {
            "hi": True,
            "time_before_responding_sec": STAGGER_LAMBDAS_WAIT_SEC
        }
        print(f"sending startup signal")
        request_ids.append(_send_one_lambda_request(data=data))

    futures = []
    executor = cf.ProcessPoolExecutor(max_workers=n)
    for request_id in request_ids:
        time.sleep(STAGGER_LAMBDA_REQUESTS_SEC)
        futures.append(executor.submit(wait_for_lambda_result, request_id=request_id))

    lambdas = {}

    for future in cf.as_completed(futures):
        # TODO: send a warming request to existing entries of `lambdas` in each round of this loop

        result, done_ts, time_sec = future.result()

        if "lambda_uid" in result:
            t = datetime.now()
            lambda_uid = result["lambda_uid"]
            lambdas[lambda_uid] = TrackedLambda(
                lambda_uid=lambda_uid, last_response_time=t
            )

            print(f"lambda {result['lambda_uid']} up in {time_sec:.2f}s")
        else:
            print(f"weirdness: didn't find lambda_uid, have data {result}")

    executor.shutdown()

    return lambdas


class LambdaPool:
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.lambdas = {}
        self.calls_in_flight = {}  # request_id --> future
        self.ml_operation_ids_to_request_ids = defaultdict(set)  # ml_operation_id --> set[request_id]
        self.request_ids_to_request_data = {}  # request_id --> dict

        self.executor = cf.ProcessPoolExecutor(max_workers=self.n_workers)

    @property
    def n_trusted(self):
        return len([l.trusted_as_warm for l in self.lambdas.values()])

    def _prune_old(self):
        lambdas = {}
        for lambda_uid, l in self.lambdas.items():
            delta = datetime.now() - l.last_response_time
            secs = delta.total_seconds()
            if secs < ASSUME_COLD_WITHIN_SECONDS:
                lambdas[lambda_uid] = l
            else:
                print(f"pruning old {lambda_uid}")
        self.lambdas = lambdas

    def _record_tracking_data(self, result, done_ts):
        if "lambda_uid" in result:
            lambda_uid = result["lambda_uid"]
            if lambda_uid in self.lambdas:
                self.lambdas[lambda_uid].last_response_time = done_ts
            else:
                self.lambdas[lambda_uid] = TrackedLambda(
                    lambda_uid=lambda_uid, last_response_time=done_ts
                )
        else:
            print(f"weirdness: didn't find lambda_uid, have data {result}")

        self._prune_old()

        # last_response_times = sorted(
        #     [l.last_response_time for l in self.lambdas.values()]
        # )
        # print(
        #     f"{len(self.lambdas)} lambdas, {self.n_trusted} trusted, last response times {last_response_times}"
        # )

    @property
    def lambdas_occupied(self) -> int:
        return sum([f.running() for f in self.calls_in_flight.values()])

    @property
    def n_requests_allowed(self) -> int:
        return self.n_workers - self.lambdas_occupied

    @property
    def all_occupied(self) -> bool:
        return self.lambdas_occupied >= self.n_workers

    def initialize(self):
        self.lambdas = secure_n_warm_lambdas(n=self.n_workers)

    def request(self, data: dict):
        ml_operation_id = str(uuid.uuid4())
        data['ml_operation_id'] = ml_operation_id

        if len(self.lambdas) == 0:
            self.initialize()

        while self.all_occupied:
            time.sleep(1)

        repeat_until_done_signal = data.get('repeat_until_done_signal', False)

        if repeat_until_done_signal:
            request_ids = request_ml_from_lambda(data=data, n_concurrent=self.n_requests_allowed)
        else:
            request_ids = [_send_one_lambda_request(data=data)]

        for request_id in request_ids:
            future = self.executor.submit(wait_for_lambda_result, request_id=request_id)
            future.recorded = False
            self.calls_in_flight[request_id] = future
            self.ml_operation_ids_to_request_ids[ml_operation_id].add(request_id)
            self.request_ids_to_request_data[request_id] = data

        return ml_operation_id

    def check(self, ml_operation_id: str):
        request_ids = self.ml_operation_ids_to_request_ids[ml_operation_id]

        if len(request_ids) == 0:
            print(f"{ml_operation_id} unknown, have {list(self.ml_operation_ids_to_request_ids.keys())}")
            return []

        futures = {request_id: self.calls_in_flight[request_id]
                   for request_id in request_ids
                   if request_id in self.calls_in_flight}

        if len(futures) == 0:
            print(f"ml_operation_id {ml_operation_id} happened, but results are missing (possibly deleted earlier)")
            return []

        collected_results = []

        for request_id, future in futures.items():
            if future.running():
                continue

            result, done_ts, time_sec = future.result()
            if not future.recorded:
                print(f"request_id {request_id} done in {time_sec:.2f}s")
                self._record_tracking_data(result, done_ts)
                future.recorded = True

            collected_results.append(result)

            request_data = self.request_ids_to_request_data[request_id]

            repeat_until_done_signal = request_data.get('repeat_until_done_signal', False)
            if repeat_until_done_signal:
                # print(f"repeat_until_done_signal --> re-requesting ml_operation_id {ml_operation_id}")
                self.request(request_data)
            # else:
            #     # not in flight anymore
            #     del self.calls_in_flight[request_id]

        return collected_results

    def done(self, ml_operation_id: str):
        to_remove = self.ml_operation_ids_to_request_ids[ml_operation_id]
        updated_calls_in_flight = {
            request_id: future
            for request_id, future in self.calls_in_flight.items()
            if request_id not in to_remove
        }
        self.calls_in_flight = updated_calls_in_flight

    def alldone(self):
        for future in self.calls_in_flight.values():
            try:
                future.cancel()
            except:
                pass
        self.calls_in_flight = {}
        self.shutdown()

    def shutdown(self):
        self.executor.shutdown()
