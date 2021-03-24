from .lambda_helpers import LambdaPool

N_CONCURRENT_LAMBDAS = 9

LAMBDA_POOL = LambdaPool(n_workers=N_CONCURRENT_LAMBDAS)
