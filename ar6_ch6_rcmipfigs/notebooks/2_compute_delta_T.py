# -*- coding: utf-8 -*-
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
# # Plot temperature response over time

# %% [markdown]
# ## IRF:
# Using forcings from RCMIP models and the impulse response function:
# \begin{align*}
# \text{IRF}(t)=& 0.885\cdot (\frac{0.587}{4.1}\cdot exp(\frac{-t}{4.1}) + \frac{0.413}{249} \cdot exp(\frac{-t}{249}))\\
# \text{IRF}(t)= &  \sum_{i=1}^2\frac{\alpha \cdot c_i}{\tau_i}\cdot exp\big(\frac{-t}{\tau_1}\big) 
# \end{align*}
# with $\alpha = 0.885$, $c_1=0.587$, $\tau_1=4.1$, $c_2=0.413$ and $\tau_2 = 249$.

# %% Thus we can estimate the mean surface temperature change from some referance year (here 0) by using [markdown]
# the estimated ERF$_x$ for some forcing agent $x$ as follows: 

# %% [markdown]
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %% [markdown]
# The ERFs are taken from models in the RCMIP [https://www.rcmip.org/](https://www.rcmip.org/)

# %% [markdown]
# ## Input data:
# See [README.md](../../README.md)

# %% [markdown]
# The data is available on request from [https://gitlab.com/rcmip/rcmip](https://gitlab.com/rcmip/rcmip). 
#

# %% [markdown]
# # Code + figures

# %% [markdown]
# ## Imports:

# %%
import xarray as xr
from IPython.display import clear_output
import numpy as np
import os
import re
from pathlib import Path
import pandas as pd
import tqdm
from scmdata import df_append, ScmDataFrame
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

# %%
from ar6_ch6_rcmipfigs.constants import BASE_DIR
from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, INPUT_DATA_DIR

PATH_DATASET = OUTPUT_DATA_DIR + '/forcing_data_rcmip_models.nc'
PATH_DT_OUTPUT = OUTPUT_DATA_DIR + '/dT_data_rcmip_models.nc'


# %% [markdown]
# ## IRF:

# %%

def IRF(t, l=0.885, alpha1=0.587 / 4.1, alpha2=0.413 / 249, tau1=4.1, tau2=249):
    """
    Returns the IRF function for:
    :param t: Time in years
    :param l: climate sensitivity factor
    :param alpha1:
    :param alpha2:
    :param tau1:
    :param tau2:
    :return:
    IRF
    """
    return l * (alpha1 * np.exp(-t / tau1) + alpha2 * np.exp(-t / tau2))


# %%

climatemodel = 'climatemodel'
scenario = 'scenario'
variable = 'variable'
time = 'time'

# %% [markdown]
# ## ERF:
# Read ERF from file

# %% [markdown]
# ### Define variables to look at:

# %%
# variables to plot:
variables_erf_comp = [
    'Effective Radiative Forcing|Anthropogenic|CH4',
    'Effective Radiative Forcing|Anthropogenic|Aerosols',
    'Effective Radiative Forcing|Anthropogenic|Tropospheric Ozone',
    'Effective Radiative Forcing|Anthropogenic|F-Gases|HFC',
    'Effective Radiative Forcing|Anthropogenic|Other|BC on Snow']
# total ERFs for anthropogenic and total:
variables_erf_tot = ['Effective Radiative Forcing|Anthropogenic',
                     'Effective Radiative Forcing']
# Scenarios to plot:
scenarios_fl = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF-aerchemmip',  
                'ssp370-lowNTCF-gidden',
                # 'ssp370-lowNTCF', Due to mistake here
                'ssp585', 'historical']

# %% [markdown]
# ## Open dataset:

# %%
ds = xr.open_dataset(PATH_DATASET)

# %% [markdown]
# # Integrate:
# The code below integrates the read in ERFs with the pre defined impulse response function (IRF).

# %% [markdown]
# \begin{align*} 
# \Delta T (t) &= \int_0^t ERF(t') IRF(t-t') dt' \\
# \end{align*}

# %%
name_deltaT = 'Delta T'


def new_varname(var, nname):
    """
    var:str
        Old variable of format varname|bla|bla
    nname:str
        name for the resulting variable, based on var
    Returns
    -------
    new variable name with nname|bla|bla
    """
    return nname + '|' + '|'.join(var.split('|')[1:])


def integrate_(i, var, nvar, ds, ds_DT, csfac=0.885):
    """
    
    Parameters
    ----------
    i:int
        the index for the integral
    var:str
        the name of the EFR variables to integrate
    nvar:str
        the name of output integrated value

    ds:xr.Dataset
        the ds with the intput data
    ds_DT: xr.Dataset
        the ouptut ds with the integrated results
    csfac: climate sensitivity factor (for IRF)
    Returns
    -------
    None
    
    """
    # lets create a ds that goes from 0 to i inclusive
    ds_short = ds[{'time': slice(0, i + 1)}].copy()
    # lets get the current year
    current_year = ds_short['time'][{'time': i}].dt.year
    # lets get a list of years
    years = ds_short['time'].dt.year
    # lets get the year delta until current year(i)
    ds_short['end_year_delta'] = current_year - years

    # lets get the irf values from 0 until i
    ds_short['irf'] = IRF(
        ds_short['end_year_delta'] * ds_short['delta_t'], l=csfac
    )

    # lets do the famous integral
    ds_short['to_integrate'] = \
        ds_short[var] * \
        ds_short['irf'] * \
        ds_short['delta_t']

    # lets sum all the values up until i and set
    # this value at ds_DT
    # If whole array is null, set value to nan
    if np.all(ds_short['to_integrate'].isnull()):  # or last_null:
        _val = np.nan
    else:
        # 

        _ds_int = ds_short['to_integrate'].sum(['time'])
        # mask where last value is null (in order to not get intgral 
        _ds_m1 = ds_short['to_integrate'].isel(time=-1)
        # where no forcing data)
        _val = _ds_int.where(_ds_m1.notnull())
    # set value in dataframe:
    ds_DT[nvar][{'time': i}] = _val


def integrate_to_dT(ds, from_t, to_t, variables, csfac=0.885):
    """
    Integrate forcing to temperature change.

    :param ds: dataset containing the focings
    :param from_t: start time
    :param to_t: end time
    :param variables: variables to integrate
    :param csfac: climate sensitivity factor
    :return:
    """
    # slice dataset
    ds_sl = ds.sel(time=slice(from_t, to_t))
    len_time = len(ds_sl['time'])
    # lets create a result DS
    ds_DT = ds_sl.copy()

    # lets define the vars of the ds
    vars = variables  # variables_erf_comp+ variables_erf_tot #['EFR']
    for var in variables:
        namevar = new_varname(var, name_deltaT)
        # set all values to zero for results dataarray:
        ds_DT[namevar] = ds_DT[var] * 0
        # Units Kelvin:
        ds_DT[namevar].attrs['unit'] = 'K'
        if 'unit' in ds_DT[namevar].coords:
            ds_DT[namevar].coords['unit'] = 'K'

    for i in range(len_time):
        # da = ds[var]
        if (i % 20) == 0:
            print('%s of %s done' % (i, len_time))
        for var in variables:
            namevar = new_varname(var, name_deltaT)  # 'Delta T|' + '|'.join(var.split('|')[1:])

            # print(var)
            integrate_(i, var, namevar, ds_sl, ds_DT, csfac=csfac)
    clear_output()

    fname = 'DT_%s-%s.nc' % (from_t, to_t)
    # save dataset.
    ds_DT.to_netcdf(fname)
    #for var in variables:
    #    ds_DT[var].plot(alpha=0.3)
    # plt.show()
    #plt.show()
    return ds_DT


csfs = [0.884, 0.526, 1.136]
csf = 0.884  # csfs[0]
# dic_ds = {}
# for csf in csfs:
_vars = variables_erf_comp + variables_erf_tot
ds_DT = integrate_to_dT(ds, '1850', '2100', _vars, csfac=csf)
# list of computed delta T variables:
variables_dt_comp = [new_varname(var, name_deltaT) for var in variables_erf_comp]

# %%
ds_DT

# %%
ds_DT.to_netcdf(PATH_DT_OUTPUT)
