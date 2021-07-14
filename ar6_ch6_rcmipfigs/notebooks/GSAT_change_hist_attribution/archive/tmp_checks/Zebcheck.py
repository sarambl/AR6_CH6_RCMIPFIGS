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
# # Check the Chapter 7 IRF
#
# Piers 

# %%
#from ar6.twolayermodel import TwoLayerModel
import numpy as np
import matplotlib.pyplot as pl
from tqdm import tqdm
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as pl
import os
from matplotlib import gridspec, rc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from netCDF4 import Dataset
import warnings
from openscm_units import unit_registry  # pip install openscm-units
from openscm_twolayermodel import ImpulseResponseModel, TwoLayerModel, constants  # pip install openscm-twolayermodel
from scmdata import ScmRun  # pip install scmdata

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
recommend_paras = pd.read_csv("../../../data_in/recommended_irf_from_2xCO2_2021_02_25_222758.csv").iloc[0, :]
recommend_paras

# %%
du = (
    recommend_paras["C (W yr / m^2 / K)"] 
    * unit_registry("W yr / m^2 / delta_degC")
    / constants.DENSITY_WATER
    / constants.HEAT_CAPACITY_WATER
).to("m")
display(du)

dl = (
    recommend_paras["C_d (W yr / m^2 / K)"] 
    * unit_registry("W yr / m^2 / delta_degC")
    / constants.DENSITY_WATER
    / constants.HEAT_CAPACITY_WATER
).to("m")
display(dl)


# %%
AR6_forcing = pd.read_csv('../../../data_in/AR6_ERF_1750-2019.csv', index_col=0)

# %%
ch6_forcing = pd.read_csv('/Users/earpmf/OneDrive - University of Leeds/PYTHON/AR6_CH6_RCMIPFIGS-fig_6.12_contrib_ERF_GSAT_hist/ar6_ch6_rcmipfigs/data_out/historic_delta_GSAT/hist_ERF_est.csv', index_col=0)

# %%
ch6_GSAT = pd.read_csv('/Users/earpmf/OneDrive - University of Leeds/PYTHON/AR6_CH6_RCMIPFIGS-fig_6.12_contrib_ERF_GSAT_hist/ar6_ch6_rcmipfigs/data_out/historic_delta_GSAT/Delta_T_timeseriies.csv', index_col=0)

# %%
ch6_forcing


# %%
ch6_GSAT

# %%
impulse_response = ImpulseResponseModel(
    d1=recommend_paras["d1 (yr)"] * unit_registry("yr"),
    d2=recommend_paras["d2 (yr)"] * unit_registry("yr"),
    q1=recommend_paras["q1 (K / (W / m^2))"] * unit_registry("delta_degC / (W / m^2)"),
    q2=recommend_paras["q2 (K / (W / m^2))"] * unit_registry("delta_degC / (W / m^2)"),
    efficacy=recommend_paras["efficacy (dimensionless)"] * unit_registry("dimensionless"),
)
res_impulse_response = impulse_response.run_scenarios()


# %%
erf = ch6_forcing['CO2']
scenario = "chapter 6 CO2"
unit = "W/m^2"

driver = ScmRun(
    data=erf,
    index=1750 + np.arange(len(erf)),
    columns={
        "unit": unit,
        "model": "custom",
        "scenario": scenario,
        "region": "World",
        "variable": "Effective Radiative Forcing",
    },
)
driver

impulse_response = ImpulseResponseModel(
    d1=recommend_paras["d1 (yr)"] * unit_registry("yr"),
    d2=recommend_paras["d2 (yr)"] * unit_registry("yr"),
    q1=recommend_paras["q1 (K / (W / m^2))"] * unit_registry("delta_degC / (W / m^2)"),
    q2=recommend_paras["q2 (K / (W / m^2))"] * unit_registry("delta_degC / (W / m^2)"),
    efficacy=recommend_paras["efficacy (dimensionless)"] * unit_registry("dimensionless"),
)
res_impulse_response = impulse_response.run_scenarios(driver)

two_layer = TwoLayerModel(
    # the naming is confusing because I follow Geoffroy, Ch. 7 Appendix does its own thing
    # the units should be clear
    lambda0=recommend_paras["alpha (W / m^2 / K)"] * unit_registry("W/m^2/delta_degC"),
    du=du,
    dl=dl,
    a=0.0 * unit_registry("watt / delta_degree_Celsius ** 2 / meter ** 2"),
    efficacy=recommend_paras["efficacy (dimensionless)"] * unit_registry("dimensionless"),
    eta=recommend_paras["kappa (W / m^2 / K)"] * unit_registry("W / m^2 / K")
)
res_two_layer = two_layer.run_scenarios(driver)

# %%
erf

# %%
res_impulse_response.head()

# %%
res_two_layer.head()

# %%
ch6_GSAT

# %%
print (0.95/1.1)

# %%
