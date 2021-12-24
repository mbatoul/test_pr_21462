# Script to test PR #21462

```sh
$ git clone https://github.com/mbatoul/test_pr_21462.git
$ cd test_pr_21462
$ conda env create -f enviromnent.yml
$ conda activate test_pr_21462
$ git clone --single-branch --branch pairwise-distances-argkmin https://github.com/jjerphan/scikit-learn.git
$ cd scikit-learn
$ python setup.py develop
$ conda list | grep scikit-learn
$ cd ..
$ python script.py
```
