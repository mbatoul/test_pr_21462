# Script to test PR #21462

Simple script to test problem benchmark results of improvements made on scikit-learn (PR #21942) against scikit-learn intelex

See benchmark [config](https://github.com/mbatoul/sklearn_benchmarks/blob/dummy_gh_pages/results/hpc_cluster/20211130T232044/config.yml) and [results](https://mbatoul.github.io/sklearn_benchmarks/results/hpc_cluster/20211130T232044/scikit_learn_intelex_vs_scikit_learn.html)

```sh
$ git clone https://github.com/mbatoul/test_pr_21462.git
$ cd test_pr_21462
$ conda env create -f environment.yml
$ conda activate test_pr_21462
$ git clone --single-branch --branch pairwise-distances-argkmin https://github.com/jjerphan/scikit-learn.git
$ cd scikit-learn
$ python setup.py develop
$ conda list | grep scikit-learn
$ cd ..
$ python script.py
```
