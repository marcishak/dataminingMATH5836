



#install: https://github.com/pymc-learn/pymc-learn#quick-install


from warnings import filterwarnings
filterwarnings("ignore")
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['THEANO_FLAGS'] = 'device=cpu'

import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(12345)
rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
      'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
sns.set(rc = rc)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#Now, let’s import the LogisticRegression model from the pymc-learn package.
import pmlearn
from pmlearn.linear_model import LogisticRegression
print('Running on pymc-learn v{}'.format(pmlearn.__version__))


#gen data
num_pred = 2
num_samples = 20000
num_categories = 2
alphas = 5 * np.random.randn(num_categories) + 5 # mu_alpha = sigma_alpha = 5
betas = 10 * np.random.randn(num_categories, num_pred) + 10 # mu_beta = sigma_beta = 10

def numpy_invlogit(x):
    return 1 / (1 + np.exp(-x))

x_a = np.random.randn(num_samples, num_pred)
y_a = np.random.binomial(1, numpy_invlogit(alphas[0] + np.sum(betas[0] * x_a, 1)))
x_b = np.random.randn(num_samples, num_pred)
y_b = np.random.binomial(1, numpy_invlogit(alphas[1] + np.sum(betas[1] * x_b, 1)))

X = np.concatenate([x_a, x_b])
y = np.concatenate([y_a, y_b])
cats = np.concatenate([ np.zeros(num_samples, dtype=np.int), np.ones(num_samples, dtype=np.int)])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, cats_train, cats_test = train_test_split(X, y, cats, test_size=0.3)



#Step 2: Instantiate a model
model = LogisticRegression()
#Step 3: Perform Inference
model.fit(X_train, y_train, cats_train, minibatch_size=200, inference_args={'n': 6000})

model.plot_elbo()

pm.traceplot(model.trace)

pm.traceplot(model.trace, lines = {"beta": betas, "alpha": alphas}, varnames=["beta", "alpha"]);

pm.summary(model.trace)