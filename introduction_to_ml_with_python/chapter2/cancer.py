import numpy as np

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("shape of cancer data: {}".format(cancer.target_names))