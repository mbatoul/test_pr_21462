

import shutil
import time
from importlib.metadata import version
from pprint import pprint

import joblib
import numpy as np
import submitit
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._show_versions import _get_deps_info, _get_sys_info
from sklearnex.neighbors import \
    KNeighborsClassifier as KNeighborsClassifierSklearnex
from threadpoolctl import threadpool_info

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

environment_information = {}
environment_information["system"] = _get_sys_info()
environment_information["dependencies"] = _get_deps_info()
environment_information["threadpool"] = threadpool_info()
environment_information["cpu_count"] = joblib.cpu_count(only_physical_cores=True)

print("Env info")
pprint(environment_information)
print("\n")

versions = {}
for lib in ["scikit-learn", "scikit-learn-intelex"]:
    versions[lib] = version(lib)

print("Versions")
pprint(versions)
print("\n")

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

print("Benchmark results")
print("speedup: ", round(speedup, 3))
print("std_speedup: ", round(std_speedup, 3))

try:
    shutil.rmtree(LOGS_FOLDER)
except OSError as e:
    print("Error: %s : %s" % (LOGS_FOLDER, e.strerror))
