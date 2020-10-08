import scipy as sp
from scipy import stats
import pandas as pd
from patsy import dmatrix
import pymc3 as pm
import theano.tensor as Tht
import matplotlib.pyplot as plt
import seaborn.apionly as sns

#==============================================================================
# data generation
#==============================================================================
labels = list('ABCDE')
true_freqs = [0.9,0.8,0.5,0.8,0.4]
Neach = 20

data = []
for label,freq in zip(labels,true_freqs):
    d = sp.rand(Neach) < freq
    data.append(pd.DataFrame(zip([label]*Neach,d.astype('int32')),
                             columns=['label','score']))

data = pd.concat(data)
Nobs = data.shape[0]
data.index = range(Nobs)

#==============================================================================
# bayesian logistic regression with pymc3
#==============================================================================
nSamples = 2000
burn = 1000
nsim = nSamples - burn

model_matrix = dmatrix(' ~ label', data=data)
x0,x1,x2,x3,x4 = [sp.array(model_matrix[:,i]).astype('int64') for i in range(5)]

with pm.Model() as model:
    # betas
    beta0 = pm.Cauchy('beta0', 0., 10.0)
    beta1 = pm.Cauchy('beta1', 0., 2.5)
    beta2 = pm.Cauchy('beta2', 0., 2.5)
    beta3 = pm.Cauchy('beta3', 0., 2.5)
    beta4 = pm.Cauchy('beta4', 0., 2.5)

    # logit
    logit_p =  (beta0*x0 + beta1*x1 + beta2*x2 + beta3*x3 + beta4*x4)
    p = Tht.exp(logit_p) / (1 + Tht.exp(logit_p))

    # likelihood
    likelihood = pm.Binomial('likelihood',n=1,p=p,observed=data['score'])

    # inference
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    trace = pm.sample(nSamples, step, progressbar=True)

### inspect
pm.traceplot(trace)

#==============================================================================
# credible intervals from posterior
#==============================================================================
# get simulated betas
'''varnames = ['beta0','beta1','beta2','beta3','beta4']
betas_mc = sp.zeros((len(varnames),nsim))
for i,varname in enumerate(varnames):
    betas_mc[i,:] = trace.get_values(varname)[nSamples-burn:]
betas_mc = sp.matrix(betas_mc)

# calculate posteriors
ps_sim = sp.zeros((Nobs,nsim))
X = sp.matrix(model_matrix)

for i in range(nsim):
    ps = X * betas_mc[:,i]
    ps_sim[:,i] = stats.logistic.cdf(ps).flatten()

#==============================================================================
# plotting
#==============================================================================
fig, axes = plt.subplots()
sns.barplot(data=data.groupby('label',as_index=False).mean(),
            x='label',y='score',ax=axes)

# calculate and add CrIs
for i, (label, group) in enumerate(data.groupby('label')):
    posterior_mc = ps_sim[group.index][0,:]
    CrIs = sp.percentile(posterior_mc,(2.5,97.5))
    axes.plot([i,i],CrIs,'k')'''
