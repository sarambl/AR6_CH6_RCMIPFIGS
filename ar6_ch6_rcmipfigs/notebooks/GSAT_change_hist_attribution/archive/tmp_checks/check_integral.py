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
# # Notebook to check integral:

# %% [markdown]
# This notebook describes how I get from ERF for CO2 to $\Delta T$ for CO2 using this equation:

# %% [markdown]
# \begin{align*}
# \Delta T_x (t) &= \int_0^t ERF_x(t') IRF(t-t') dt' \\
# \end{align*}
# %% [markdown]
# #### The Impulse response function (IRF):
# In these calculations we use:
# \begin{align*}
# IRF(t) = \frac{q_1}{d_1} \exp\Big(\frac{-t}{d_1}\Big) + \frac{q_2}{d_2} \exp\Big(\frac{-t}{d_2}\Big)
# \end{align*}
#
# Where the constants, $q_i$ and $d_i$ are shown below. 
#
#
# %%
import pandas as pd
import xarray as xr
from IPython.display import clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR

# %% [markdown]
# ### Get IRF parameters 

# %%
fn_IRF_constants = INPUT_DATA_DIR /'recommended_irf_from_2xCO2_2021_02_25_222758.csv'

irf_consts = pd.read_csv(fn_IRF_constants).set_index('id')

ld1 = 'd1 (yr)'
ld2 = 'd2 (yr)'
lq1 = 'q1 (K / (W / m^2))'
lq2 = 'q2 (K / (W / m^2))'
median = 'median'
perc5 = '5th percentile'
perc95 = '95th percentile'
recommendation = 'recommendation'
irf_consts  # [d1]

# %%
# lets get the irf values from 0 until i
d1 = float(irf_consts[ld1])
d2 = float(irf_consts[ld2])
q1 = float(irf_consts[lq1])
q2 = float(irf_consts[lq2])

print(f'd1={d1}, d2={d2}, q1={q1}, q2={q2}')


# %% [markdown]
# ### Path input data ERF

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

#PATH_DATASET = OUTPUT_DATA_DIR / 'ERF_data.nc'
PATH_DATASET = OUTPUT_DATA_DIR/'historic_delta_GSAT/hist_ERF_est.csv'


# %% [markdown]
# ### Open ERF dataset:

# %%
df = pd.read_csv(PATH_DATASET, index_col=0)
da_ERF = df.to_xarray().to_array()#'variable'
da_ERF = da_ERF.rename({'index':'year'})
#ds = xr.open_dataset(PATH_DATASET).sel(year=slice(1700, 2200))  # we need only years until 1700
ds = xr.Dataset({'ERF':da_ERF})
ds
#da_ERF = ds['ERF']

# %%
# name of output variable
name_deltaT = 'Delta T'

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'
percentile = 'percentile'

# %%
ds['ERF'].to_pandas().transpose().to_csv('ERF_timeseries.csv')

# %% [markdown]
# ### Simple pre-processing

# %%
ds['time'] = pd.to_datetime(ds['year'].to_pandas().index.map(str), format='%Y')

# delta_t is 1 (year)
ds['delta_t'] = xr.DataArray(np.ones(len(ds['year'])), dims='year', coords={'year': ds['year']})
ds['year']


# %% [markdown]
# # Integrate from ERF to delta T
#
#

# %% [markdown]
# ## IRF function: 

# %%

def IRF(t, d1, q1, d2, q2):
    """
    Returns the IRF function for:
    :param q2:
    :param d2:
    :param q1:
    :param d1:
    :param t: Time in years
    :return:
    IRF
    """
    #print(f'd1={d1}, d2={d2}, q1={q1}, q2={q2}')
    irf = q1 / d1 * np.exp(-t / d1) + q2 / d2 * np.exp(-t / d2)
    return irf
    # l * (alpha1 * np.exp(-t / tau1) + alpha2 * np.exp(-t / tau2))


# %% [markdown]
#
#
# Calculate for 2019, meaning integrating over i years:

# %%
i = 2019-1750


# %% [markdown]
# ## select only CO2 for check:

# %%
ds_in = ds.copy().sel(variable='CO2')

# %% [markdown]
# ### slice data to contain only years before year i

# %%
erf ='ERF'
ds_short = ds_in[{'year': slice(0, i + 1)}].copy()
len(ds_short['ERF'])+1750

# %%
# lets get the current year
current_year = ds_short['year'][{'year': i}]  # .dt.year
current_year

# %% [markdown]
# ### IRF is calculated for t-t': calculates t-t'

# %%
# lets get a list of years
_years = ds_short['year']  # .dt.year
# lets get the year delta until current year(i)
ds_short['end_year_delta'] = current_year - _years +.5
print(ds_short)

# %% [markdown]
# ### Calculate IRF:

# %%
ds_short['irf'] = IRF(
    # t-t':
    ds_short['end_year_delta'] * ds_short['delta_t'], 
    # parameters
    d1, q1, d2, q2)


# %% [markdown]
# # The integrand:
#

# %% [markdown]
# \begin{align*}
#  ERF_x(t') IRF(t-t') \cdot \Delta t' 
# \end{align*}
# %%
# lets do the famous integral
ds_short['to_integrate'] = \
        ds_short[erf] * \
        ds_short['irf'] * \
        ds_short['delta_t']

# %%
ds_short

# %% [markdown]
# ### Finally do the sum:

# %%
ds_short['to_integrate'].sum()

# %%
q2

# %%

# %% [markdown]
# (a+da)(db)*dt =

# %% [markdown]
# (a+da)(b+db)-a*b != 

# %% [markdown]
# $f(x) = x^2$

# %% [markdown]
# $\int_0^2 f dx$

# %% [markdown]
# 1*2=2

# %% [markdown]
# $\frac{1}{3}x^3= \frac{1}{3}8=2.66$ 

# %%
8/3

# %%
