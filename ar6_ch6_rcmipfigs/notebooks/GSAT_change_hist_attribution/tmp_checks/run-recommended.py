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

# %%
import numpy as np
import pandas as pd
from openscm_units import unit_registry  # pip install openscm-units
from openscm_twolayermodel import ImpulseResponseModel, TwoLayerModel, constants  # pip install openscm-twolayermodel
from scmdata import ScmRun  # pip install scmdata

# %%
recommend_paras = pd.read_csv("recommended_irf_from_2xCO2_2021_02_25_222758.csv").iloc[0, :]
recommend_paras

# %%
erf = np.arange(200) * 4 / 70
scenario = "1pctCO2"
unit = "W/m^2"

driver = ScmRun(
    data=erf,
    index=1850 + np.arange(len(erf)),
    columns={
        "unit": unit,
        "model": "custom",
        "scenario": scenario,
        "region": "World",
        "variable": "Effective Radiative Forcing",
    },
)
driver

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
res.filter(variable="Surface Temperature*").lineplot(
    hue="climate_model", style="variable"
)
