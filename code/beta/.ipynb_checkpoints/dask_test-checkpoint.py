import numpy as np
import dill
import joblib
# from distributed import Client
from dask_jobqueue import SLURMCluster
from joblib import Parallel, delayed
time_stamp = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")

cluster = SLURMCluster(cores=8,
                       processes=1,
                       memory="16GB",
                       walltime="01:00:00",
                       queue="carney-sjones-condo")