#
# Simple script to test problems results with:
#   https://github.com/mbatoul/sklearn_benchmarks/blob/dummy_gh_pages/results/hpc_cluster/20211130T232044/config.yml
#   https://mbatoul.github.io/sklearn_benchmarks/results/hpc_cluster/20211130T232044/scikit_learn_intelex_vs_scikit_learn.html
#
# Setup env:
# conda create -n test_pr_21462 -c conda-forge scikit-learn-intelex submitit cython numpy scipy
# conda activate test_pr_21462
# conda remove scikit-learn
# git clone --single-branch --branch pairwise-distances-argkmin https://github.com/jjerphan/scikit-learn.git
# cd scikit-learn
# python setup.py develop
# conda list | grep scikit-learn
# cd ..
# python script_pr_21462.py

import submitit
import time
import shutil

import numpy as np
from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier
from sklearnex.neighbors import KNeighborsClassifier as KNeighborsClassifierSklearnex

N_EXECUTIONS = 10
LOGS_FOLDER = "./tmp"


def run_bench_one_func(func, *args):
    t0 = time.perf_counter()
    res = func(*args)
    t1 = time.perf_counter()

    return t1 - t0


def run_benchmark(estimator):
    estimator.fit(X_train, y_train)

    times = []
    jobs = [
        executor.submit(run_bench_one_func, estimator.predict, X_test)
        for _ in range(N_EXECUTIONS)
    ]
    times = [job.result() for job in jobs]

    mean_duration = np.mean(times)
    std_duration = np.std(times)

    return mean_duration, std_duration


executor = submitit.AutoExecutor(folder=LOGS_FOLDER)
executor.update_parameters(
    timeout_min=70,
    slurm_partition="parietal",
    cpus_per_task=16,
    slurm_additional_parameters=dict(hint="nomultithread"),
)

# Matches part of the config given above
X_train, y_train = make_classification(
    n_samples=100_000, n_classes=2, n_redundant=0, n_features=100
)
X_test, y_test = make_classification(
    n_samples=1000, n_classes=2, n_redundant=0, n_features=100
)

# sklearn
knn_sklearn = KNeighborsClassifier(algorithm="brute")
mean_duration_sklearn, std_duration_sklearn = run_benchmark(knn_sklearn)

# sklearnex
knn_sklearnex = KNeighborsClassifierSklearnex(algorithm="brute")
mean_duration_sklearnex, std_duration_sklearnex = run_benchmark(knn_sklearnex)

speedup = mean_duration_sklearn / mean_duration_sklearnex
std_speedup = speedup * (
    np.sqrt(
        (std_duration_sklearn / mean_duration_sklearn) ** 2
        + (std_duration_sklearnex / mean_duration_sklearnex) ** 2
    )
)
print("speedup: ", round(speedup, 3))
print("std_speedup: ", round(std_speedup, 3))

try:
    shutil.rmtree(LOGS_FOLDER)
except OSError as e:
    print("Error: %s : %s" % (LOGS_FOLDER, e.strerror))