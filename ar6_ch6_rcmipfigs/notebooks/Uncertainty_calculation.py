# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Error bars in plots:

# %% [markdown]
# ### Sources of uncertainty:
# In these calculations we are considering the following uncertainties
# 1. Model uncertainty
# 2. IRF uncertainty/climate sensitivity uncertainty
#
# Model uncertainty is represented as the spread in the ERF produced by the considered RCMIP models. IRF uncertainty is the uncertainty by which the ERF is translated into changes in temperature. 
#
#

# %% [markdown]
# ## IRF:
# In these calculations we use the impulse response function:
# \begin{align*}
# \text{IRF}(t)=& 0.885\cdot (\frac{0.587}{4.1}\cdot exp(\frac{-t}{4.1}) + \frac{0.413}{249} \cdot exp(\frac{-t}{249}))\\
# \text{IRF}(t)= &  \sum_{i=1}^2\frac{\alpha \cdot c_i}{\tau_i}\cdot exp\big(\frac{-t}{\tau_1}\big) 
# \end{align*}
# with $\alpha = 0.885$, $c_1=0.587$, $\tau_1=4.1$, $c_2=0.413$ and $\tau_2 = 249$.

# %% Thus we can estimate the mean surface temperature change from some referance year (here 0) by using [markdown]
# ### Calculate $\Delta T$ from ERF:
# We then use the estimated ERF$_x$ for some forcing agent(s) $x$ as follows: 

# %% [markdown]
# \begin{align*} 
# \Delta T_x (t) &= \int_0^t ERF_x(t') IRF(t-t') dt' \\
# \end{align*}

# %% [markdown]
# Now, define $\Delta_x$ as follows:
# \begin{align}
# \Delta_x  = & \frac{1}{\alpha} \int_0^t ERF_x(t') IRF(t-t') dt'\\
#  =& \frac{1}{\alpha} \int_0^t ERF_x(t')   \sum_{i=1}^2\frac{\alpha \cdot c_i}{\tau_i}\cdot exp\big(\frac{-(t-t')}{\tau_1}\big)dt' \\
#  =&  \int_0^t ERF_x(t') \sum_{i=1}^2\frac{c_i}{\tau_i}\cdot exp\big(\frac{-(t-t')}{\tau_1}\big)dt' \\
# \end{align}

# %% [markdown]
# So, then: 
# \begin{align}
# \Delta T_x (t) = \alpha \cdot \Delta_x(t)
# \end{align}

# %% [markdown]
# This means that the uncertainty in $\Delta T$ can be calculated according to the propagated uncertainty in the product of parameter $\alpha$ and uncertainty in ERF$_x$.

# %% [markdown]
# ### Distribution of a product of two independent variables:
# Assuming these two are independent we get:
# \begin{align}
# Var(\Delta T_x) = &Var(\alpha\cdot \Delta_{x})\\
#  = & (Var(\alpha) +E(\alpha)^2)(Var(\Delta_{x}) + E( \Delta_{x})^2) - E(\alpha)^2E(\Delta_{x})^2
# \end{align}

# %% [markdown]
# Let $\sigma_x= \sqrt{Var(\Delta_{x})}$, $\mu_x=  E(\Delta_{x})$, $\sigma_\alpha = \sqrt{Var(\alpha)}$ and $\mu_\alpha = E(\alpha)$ 

# %% [markdown]
# \begin{align}
# Var(\Delta T_x) = (\sigma_x^2 + \mu_x^2)(\sigma_\alpha^2+\mu_\alpha^2) - \mu_x^2 \mu_\alpha^2
# \end{align}

# %% [markdown]
# ## Method:

# %% [markdown]
# The following method is used:
# 1. Intra model variability from $ERF$ from different models
# 2. Assume this is independent of the $IRF$
# 3. Combine these two uncertainties with $Var(\Delta T_x) = (\sigma_x^2 + \mu_x^2)(\sigma_\alpha^2+\mu_\alpha^2) - \mu_x^2 \mu_\alpha^2$

# %% [markdown]
# ## Sums and differences:
# For any additive combination of several components (either sum of two SLCF's or difference etc), e.g. the difference between methane contribution $X_i$ and the total anthropogenic contribution $Y$, we would have some covariance between between X and Y, because if one model has large $X_i$ it would normally have large $Y$ as well. 
# So either we can take this into account explicitly:
# $$ Var(X+Y) = Var(X)+Var(Y) +2Cov(X,Y)$$
# Alternatively, we can treat the sum or difference of the ERF as one stocastic variable and alpha as another and assume they are independent. The independence of the error on ECS and ERF is a good assumption here. Secondly, we do then not need to consider the covariance of ERF between different components because it is implicitly covered. 
#
#
# ### Summary: 
# Let $\sigma_{\alpha}$ and $\mu_{\alpha}$ be the standard deviation and mean for a normal distribution of the $\alpha$ parameter in ECS. Secondly, let $X_i$ be a sample of 
#

# %% [markdown]
# \begin{align}
# X_i  = & \frac{1}{\alpha} \int_0^t ERF_i(t') IRF(t-t') dt'\\
#  =&  \int_0^t ERF_i(t') \sum_{i=1}^2\frac{c_i}{\tau_i}\cdot exp\big(\frac{-(t-t')}{\tau_1}\big)dt' \\
# \end{align}
# where $ERF_i$ is some difference or sum of different ERF components. 

# %% [markdown]
# Then 
# \begin{align}
# \sigma_{X_i} = \sqrt{\frac{\sum(X_{i,k}-\mu_{X_i})^2}{N}}
# \end{align}

# %% [markdown]
# and we can get 
# \begin{align}
# \sigma_T = (\sigma_{X_i}+\mu_{X_i})(\sigma_{\alpha} + \mu_{\alpha}) - \mu_{X_i}\mu_{\alpha}
# \end{align}

# %% [markdown]
# ### Technical calculation:
# From any calculation of 
# \begin{align}
# \Delta T_{\alpha=\mu_\alpha} = \sum_i T_i - \sum_k T_k
# \end{align}
# for all models, calculated with IRF such that $\alpha = \mu_{\alpha}$, we can find 
# \begin{align}
# X_{i,k} = \frac{1}{\mu_{\alpha}} \Delta T_{\alpha=\mu_\alpha,k}
# \end{align}
# where the index $k$ signifies the different models. 
#
# And thus we can easily calculate
# \begin{align}
# \sigma_{X_i} = \sqrt{\frac{\sum(X_{i,k}-\mu_{X_i})^2}{N}}
# \end{align}

# %% [markdown]
# since 
# \begin{align}
# \mu_{X_i} = \frac{1}{\mu_\alpha}\mu_{\Delta T_{\alpha=\mu_\alpha}}
# \end{align}
# we have
# \begin{align}
# \sigma_{X_i} = \frac{1}{\mu_\alpha} \sigma_{\Delta T_{\alpha=\mu_\alpha}}.
# \end{align}

# %% [markdown]
# ## Finally:
# Let $\Delta T = X_{i}\cdot \alpha $ and assume $X_i$ and $\alpha$ independent. 
# Then
# \begin{align}
# \sigma_{\Delta T}^2 =& (\sigma_{X_i}^2+\mu_{X_i}^2)(\sigma_{\alpha}^2 + \mu_{\alpha}^2) - \mu_{X_i}^2\mu_{\alpha}^2\\
# \sigma_{\Delta T}^2 =& \frac{1}{\mu_\alpha^2}\big[(\sigma_{\Delta T_{\alpha=\mu_\alpha} }^2 +\mu_{\Delta T_{\alpha=\mu_\alpha}}^2)(\sigma_{\alpha}^2 + \mu_{\alpha}^2) - \mu_{\Delta T_{\alpha=\mu_\alpha}}^2\mu_{\alpha}^2 \big]\\
# \sigma_{\Delta T} =& \frac{1}{\mu_\alpha}\big[(\sigma_{\Delta T_{\alpha=\mu_\alpha} }^2 +\mu_{\Delta T_{\alpha=\mu_\alpha}}^2)(\sigma_{\alpha}^2 + \mu_{\alpha}^2) - \mu_{\Delta T_{\alpha=\mu_\alpha}}^2\mu_{\alpha}^2 \big]^{\frac{1}{2}}
# \end{align}
#

# %%
def sigma_DT(dT, sig_alpha, mu_alpha, dim='climatemodel'):
    sig_DT = dT.std(dim)
    mu_DT = dT.mean(dim)
    return ((sig_DT**2 + mu_DT**2)*(sig_alpha**2+mu_alpha**2)- mu_DT**2*mu_alpha**2)**(0.5)/mu_alpha

# %% [markdown]
# In other words, it suffices to know 
#
# a) $\sigma_\alpha$ and $\mu_\alpha$ and 
#
# b) $\Delta T_x$ calculated for a fixed $\mu_\alpha$ 
#
# to compute the uncertainty bars. 
