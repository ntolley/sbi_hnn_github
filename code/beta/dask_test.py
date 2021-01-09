import numpy as np
import dill
import joblib
from distributed import Client
from dask_jobqueue import SLURMCluster
import dask
from joblib import Parallel, delayed
import time
import random
import pandas as pd
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

cluster = SLURMCluster(
                       cores=8,
                       processes=1,
                       memory="20GB",
                       walltime="01:00:00",
                    #    queue="carney-sjones-condo"
)

print(cluster.job_script())
cluster.scale(2)
client = Client(cluster)



def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)

input_params = pd.DataFrame(np.random.random(size=(500, 4)),
                            columns=['param_a', 'param_b', 'param_c', 'param_d'])

lazy_results = []

for parameters in input_params.values[:10]:
    lazy_result = dask.delayed(costly_simulation)(parameters)
    lazy_results.append(lazy_result)

dask.compute(*lazy_results)