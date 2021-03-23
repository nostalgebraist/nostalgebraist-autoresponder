from experimental.lambda_helpers import LambdaPool

N_CONCURRENT_LAMBDAS = 2

LAMBDA_POOL = LambdaPool(n_workers=N_CONCURRENT_LAMBDAS)
