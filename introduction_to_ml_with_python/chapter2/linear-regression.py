import mglearn
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = Lasso(alpha=0.0001).fit(X_train, y_train)
print("lr.conf: {}".format(lr.coef_))
print("lr.score: {}".format(lr.score(X_train, y_train)))
print("lr.score: {}".format(lr.score(X_test, y_test)))

