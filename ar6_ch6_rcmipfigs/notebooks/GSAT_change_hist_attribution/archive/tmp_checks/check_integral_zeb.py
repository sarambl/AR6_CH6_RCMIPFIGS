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
import scipy.integrate

from openscm_units import unit_registry  # pip install openscm-units
from openscm_twolayermodel import ImpulseResponseModel, TwoLayerModel, constants  # pip install openscm-twolayermodel
from scmdata import ScmRun  # pip install scmdata

# %load_ext autoreload
# %autoreload 2
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR, OUTPUT_DATA_DIR

# %% [markdown]
# ### Get IRF parameters 

# %%
# fn_IRF_constants = INPUT_DATA_DIR /'recommended_irf_from_2xCO2_2021_02_25_222758.csv'
fn_IRF_constants = 'recommended_irf_from_2xCO2_2021_02_25_222758.csv'

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

# %% jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
# from ar6_ch6_rcmipfigs.constants import OUTPUT_DATA_DIR, RESULTS_DIR

# PATH_DATASET = OUTPUT_DATA_DIR/'historic_delta_GSAT/hist_ERF_est.csv'
#PATH_DATASET = "test_erf.csv"
PATH_DATASET = OUTPUT_DATA_DIR/'historic_delta_GSAT/hist_ERF_est.csv'


# %% [markdown]
# ### Open ERF dataset:

# %%
df = pd.read_csv(PATH_DATASET, index_col=0)
da_ERF = df.to_xarray().to_array()#'variable'
da_ERF = da_ERF.rename({'index':'year'})
#ds = xr.open_dataset(PATH_DATASET).sel(year=slice(1700, 2200))  # we need only years until 1700
ds = xr.Dataset({'ERF':da_ERF})
ds#['ERF']
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
ds


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
i

# %% [markdown]
# ## select only CO2 for check:

# %%
ds_in = ds.copy().sel(variable='CO2')
ds_in

# %% [markdown]
# ### slice data to contain only years before year i

# %%
erf ='ERF'
ds_short = ds_in[{'year': slice(0, i + 1)}].copy()
ds_short

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
ds_short['end_year_delta'] = current_year - _years
print(ds_short)

# %% [markdown]
# ### Calculate IRF:

# %%
ds_short['irf'] = IRF(
    # t-t':
    ds_short['end_year_delta'] * ds_short['delta_t'], 
    # parameters
    d1, q1, d2, q2)
ds_short['irf']

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


# %% [markdown]
# ## Do the sum without a Riemann approximation

# %%
def combo_integrand(t, d1, q1, d2, q2, erf, end_year):
    erf_interp = np.interp(t, erf["year"].values, erf.values)
#     print(erf_interp)
    delta_t = end_year - t
#     print(delta_t)
    irf = IRF(delta_t, d1, q1, d2, q2)
#     print(irf)
    integrand = erf_interp * irf
    
    return integrand


# %%
end_year = 2019
tmp = scipy.integrate.quad(combo_integrand, 1750, end_year, args=(d1, q1, d2, q2, ds_short[erf], end_year), limit=500)
exact_integration_result = tmp[0]
exact_integration_result

# %%
# double check that it is just the difference because of the Riemann approximation,
# not sum silly coding error
riemann_approx = combo_integrand(np.arange(1750, 2019 + 1), d1, q1, d2, q2, ds_short[erf], end_year).sum()
np.testing.assert_allclose(riemann_approx, ds_short['to_integrate'].sum())
riemann_approx

# %%
plt.plot(ds_short['to_integrate']['year'].values[-10:], ds_short['to_integrate'].values[-10:])
plt.step(ds_short['to_integrate']['year'].values[-10:], ds_short['to_integrate'].values[-10:])

# %% [markdown]
# ## Compare with openscm-twolayermodel

# %%
recommend_paras = pd.read_csv("../../../data_in/recommended_irf_from_2xCO2_2021_02_25_222758.csv").iloc[0, :]
erf = pd.read_csv(PATH_DATASET, index_col=0)
scenario = "test"
unit = "W/m^2"

driver = ScmRun(
    data=erf["CO2"].values,
    index=erf.index.values,
    columns={
        "unit": unit,
        "model": "custom",
        "scenario": scenario,
        "region": "World",
        "variable": "Effective Radiative Forcing",
    },
)
du = (
    recommend_paras["C (W yr / m^2 / K)"] 
    * unit_registry("W yr / m^2 / delta_degC")
    / constants.DENSITY_WATER
    / constants.HEAT_CAPACITY_WATER
).to("m")

dl = (
    recommend_paras["C_d (W yr / m^2 / K)"] 
    * unit_registry("W yr / m^2 / delta_degC")
    / constants.DENSITY_WATER
    / constants.HEAT_CAPACITY_WATER
).to("m")

two_layer = TwoLayerModel(
    # the naming is confusing because I follow Geoffroy, Ch. 7 Appendix does its own thing
    # the units should be clear
    lambda0=recommend_paras["alpha (W / m^2 / K)"] * unit_registry("W/m^2/delta_degC"),
    du=du,
    dl=dl,
    a=0.0 * unit_registry("watt / delta_degree_Celsius ** 2 / meter ** 2"),
    efficacy=recommend_paras["efficacy (dimensionless)"] * unit_registry("dimensionless"),
    eta=recommend_paras["kappa (W / m^2 / K)"] * unit_registry("W / m^2 / K"),
#     delta_t=100 * unit_registry("yr"),
)
# res_two_layer = two_layer.run_scenarios(driver.interpolate([dt.datetime(y, m, 1) for y in 1850 + np.arange(len(erf)) for m in range(1, 13)]))
res_two_layer = two_layer.run_scenarios(driver)

impulse_response = ImpulseResponseModel(
    d1=recommend_paras["d1 (yr)"] * unit_registry("yr"),
    d2=recommend_paras["d2 (yr)"] * unit_registry("yr"),
    q1=recommend_paras["q1 (K / (W / m^2))"] * unit_registry("delta_degC / (W / m^2)"),
    q2=recommend_paras["q2 (K / (W / m^2))"] * unit_registry("delta_degC / (W / m^2)"),
    efficacy=recommend_paras["efficacy (dimensionless)"] * unit_registry("dimensionless"),
)
res_impulse_response = impulse_response.run_scenarios(driver)

res = res_two_layer.append(res_impulse_response)
res.head()

# %%
res_impulse_response.head()

# %%
res_openscm_twolayer = res.filter(variable="Surface Temperature", year=2019).timeseries(meta=["climate_model", "scenario", "region", "variable", "unit"])
np.testing.assert_allclose(res_openscm_twolayer, exact_integration_result, atol=0.01)
display(exact_integration_result)
display(res_openscm_twolayer)
