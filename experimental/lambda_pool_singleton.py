from .lambda_helpers import LambdaPool

N_CONCURRENT_LAMBDAS = 1

LAMBDA_POOL = LambdaPool(n_workers=N_CONCURRENT_LAMBDAS)
