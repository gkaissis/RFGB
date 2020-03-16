#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from collections import Counter
import numpy as np

np.set_printoptions(precision=3)
import jax
import jax.numpy as jnp
from typing import Tuple, Union, List


# # 1. Datasets

# In[2]:


# Classification Dataset
X_class, y_class = load_breast_cancer(return_X_y=True)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class)

### Regression Dataset
X_reg, y_reg = load_boston(return_X_y=True)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg)


# # 2. Random Forest

# In[3]:


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        subsample: float = 0.1,
        regression: bool = True,
        **kwargs
    ) -> None:
        self.n_estimators = n_estimators
        self.estimators = []  # type:ignore
        self.subsample = subsample

        for _ in range(self.n_estimators):
            if regression:
                self.estimators.append(DecisionTreeRegressor(**kwargs))
            else:
                self.estimators.append(DecisionTreeClassifier(**kwargs))

    def bootstrap_sample(self, X: np.array, y: np.array) -> Tuple[np.array, ...]:
        n_samples = X.shape[0]
        idxs = np.random.choice(
            n_samples, size=int(self.subsample * n_samples), replace=True
        )
        return X[idxs], y[idxs]

    def fit(self, X: np.array, y: np.array) -> None:
        for estimator in self.estimators:
            X_sample, y_sample = self.bootstrap_sample(X, y)
            estimator.fit(X_sample, y_sample)

    def predict(self, X: np.array) -> np.array:
        preds = np.array([estimator.predict(X) for estimator in self.estimators])
        preds = np.swapaxes(preds, 0, 1)
        return np.array([self._most_common_pred(pred) for pred in preds])

    def _most_common_pred(self, y: np.array) -> Union[float, int]:
        return Counter(y).most_common(1)[0][0]


# ## 3.1. Random Forest Classifier

# In[4]:


rfc = RandomForest(regression=False)


# In[5]:


rfc.fit(X_train_c, y_train_c)


# In[19]:


print(accuracy_score(rfc.predict(X_test_c), y_test_c).round(3))


# ## 3.2. Random Forest Regressor

# In[7]:


rfr = RandomForest(regression=True)


# In[8]:


rfr.fit(X_train_r, y_train_r)


# In[20]:


print(mean_absolute_error(y_test_r, rfr.predict(X_test_r)).round(3))


# ___
#
# # 4. Gradient Boosting

# In[10]:


# Loss functions
def MSE(y_true: jnp.array, y_pred: jnp.array):
    return jnp.mean(jnp.sum(jnp.square(y_true - y_pred)))


def CrossEntropy(y_true: jnp.array, y_proba: jnp.array):
    y_proba = jnp.clip(y_proba, 1e-5, 1 - 1e-5)
    return jnp.sum(-y_true * jnp.log(y_proba) - (1 - y_true) * jnp.log(1 - y_proba))


# In[11]:


class GradientBoosting:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        regression: bool = True,
        **kwargs
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.regression = regression
        self.loss = MSE if self.regression else CrossEntropy

        self.estimators = []  # type:ignore
        for _ in range(self.n_estimators):
            self.estimators.append(DecisionTreeRegressor(**kwargs))

    def fit(self, X: np.array, y: np.array):
        y_pred = np.full(np.shape(y), np.mean(y))
        for i, estimator in enumerate(self.estimators):
            gradient = jax.grad(self.loss, argnums=1)(
                y.astype(np.float32), y_pred.astype(np.float32)
            )
            self.estimators[i].fit(X, gradient)
            update = self.estimators[i].predict(X)
            y_pred -= self.learning_rate * update

    def predict(self, X: np.array):
        y_pred = np.zeros(X.shape[0], dtype=np.float32)
        for estimator in self.estimators:
            y_pred -= self.learning_rate * estimator.predict(X)

        if not self.regression:
            return np.where(np.exp(y_pred) > 1, 1, 0)
        return y_pred


# ## 4.1 Gradient Boosting Regressor

# In[12]:


gbr = GradientBoosting(regression=True)


# In[13]:


gbr.fit(X_train_r, y_train_r)


# In[21]:


print(mean_absolute_error(gbr.predict(X_test_r), y_test_r).round(3))


# ## 4.2 Gradient Boosting Classifier

# In[15]:


gbc = GradientBoosting(regression=False)


# In[16]:


gbc.fit(X_train_c, y_train_c)


# In[22]:


print(accuracy_score(gbc.predict(X_test_c), y_test_c).round(3))

