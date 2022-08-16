# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Derive emission based historical ERFs:
# We use timeseries for historical emissions/concentrations and then scale these with the 1750-2019 ERF from Thornhill (2019)/bill collins plot to derive ERF timeseries. 
# For short lived components, we use change in emissions from CEDS:
# - NOx, VOC/CO, SO2, OC, BC, NH3
#
# For longer lived components, we use change in concentrations from chap 2:
# - CO2, CH4, N2O, HC
#
# For ERF from VOC/CO we use emissions of CO to scale. 
# For HC we use the HC from Thornhill (which includes only CFCs and HCFCs) and additionally, we use HFC concentrations (ch2) and multipy with radiative efficiencies (RE) from Hodnebrog et al (2019). 
# HC is then the sum of the HC from Thornhill (scaled with concentrations) and the HFC ERF.
#
# Finally, these ERFs are integrated with the IRF and the change in GSAT is calculated. 
#

# %% [markdown]
# ### References:
# Hodnebrog, Ø, B. Aamaas, J. S. Fuglestvedt, G. Marston, G. Myhre, C. J. Nielsen, M. Sandstad, K. P. Shine, and T. J. Wallington. “Updated Global Warming Potentials and Radiative Efficiencies of Halocarbons and Other Weak Atmospheric Absorbers.” Reviews of Geophysics 58, no. 3 (2020): e2019RG000691. https://doi.org/10.1029/2019RG000691
#
#
# Thornhill, Gillian D., William J. Collins, Ryan J. Kramer, Dirk Olivié, Ragnhild B. Skeie, Fiona M. O’Connor, Nathan Luke Abraham, et al. “Effective Radiative Forcing from Emissions of Reactive Gases and Aerosols – a Multi-Model Comparison.” Atmospheric Chemistry and Physics 21, no. 2 (January 21, 2021): 853–74. https://doi.org/10.5194/acp-21-853-2021.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR_BADC, INPUT_DATA_DIR, \
    OUTPUT_DATA_DIR, RESULTS_DIR
from ar6_ch6_rcmipfigs.utils.badc_csv import read_csv_badc

# %% [markdown]
# ### File paths

# %%


# %%
fn_concentrations = INPUT_DATA_DIR / 'historical_delta_GSAT/LLGHG_history_AR6_v9_updated.xlsx'
path_emissions = INPUT_DATA_DIR / 'historical_delta_GSAT/CEDS_v2021-02-05_emissions/'

# file path table of ERF 2019-1750
fp_collins = OUTPUT_DATA_DIR / 'fig6_12_ts15_historic_delta_GSAT/table_mean_thornhill_collins_orignames.csv'

# %%
fl_CEDS = list(path_emissions.glob('*global_CEDS_emissions_by_sector_2021_02_05.csv'))

# %% [markdown]
# ### Output file paths:

# %%
fn_output_ERF = OUTPUT_DATA_DIR / 'fig6_12_ts15_historic_delta_GSAT/hist_ERF_est.csv'
fn_output_ERF_2019 = OUTPUT_DATA_DIR / 'fig6_12_ts15_historic_delta_GSAT/2019_ERF_est.csv'
fn_output_decomposition = OUTPUT_DATA_DIR / 'fig6_12_ts15_historic_delta_GSAT/hist_ERF_est_decomp.csv'

# %% [markdown]
# ## Load concentration file and interpolate from 1750 to 1850

# %%

df_conc = pd.read_excel(fn_concentrations, header=22, index_col=0, engine='openpyxl')
# adds unnecessary row with nans and unnamed columns
df_conc = df_conc.loc[1750:2019]
unnamed = [c for c in df_conc.columns if 'Unnamed:' in c]
df_conc = df_conc.drop(unnamed, axis=1)

# For C8F18 there appears to be an error in the spreadsheet where 2015 is entered as zero, presumably 0.09 but treat as missing
df_conc.loc[2015, 'C8F18'] = np.nan

# datetime index
df_conc.index = pd.to_datetime(df_conc.index.astype(int), format='%Y')

# resample to yearly, i.e. NaNs will be filled between 1750 and 1850:
df_conc = df_conc.resample('Y').first()  # .interpolate()
# Interpolate:
df_conc = df_conc.interpolate(method='linear',
                              limit_area='inside')  # 'inside' only fill values with valid on both sides.
# reset index to year
df_conc.index = df_conc.index.year
df_conc

# %%
[c for c in df_conc.columns if 'CFC' in c]

# %% [markdown]
# ## Emissions:

# %%
list_df_em = []
units_dic = {}
for fn in fl_CEDS:
    _df = pd.read_csv(fn)
    u_em = _df['em'].unique()
    if len(u_em) > 1:
        print('double check, emission labels :')
        print(u_em)
    _em = u_em[0]
    u_units = _df['units'].unique()
    if len(u_units) > 1:
        print('double check, units labels :')
        print(u_units)
    _un = u_units[0]
    _df_s = _df.drop(['em', 'sector', 'units'], axis=1).sum()
    _df_s.index = pd.to_datetime(_df_s.index, format='X%Y').year
    _df = pd.DataFrame(_df_s, columns=[_em])
    units_dic[_em] = _un
    list_df_em.append(_df)

# %%
df_emissions = pd.concat(list_df_em, axis=1)

# %%
df_emissions

# %%
units_dic

# %% [markdown]
# ## Load CMIP ERFs (bill collins)

# %%
df_collins = pd.read_csv(fp_collins, index_col=0)
df_collins.index = df_collins.index.rename('emission_experiment')

# %%
df_collins  # .columns

# %%
df_collins.sum()  # .columns

# %%
forcing_total_collins = df_collins.sum(axis=1)  # ['Total']
forcing_total_collins

# %%


# %%
def scale_ERF(forcing_tot, df_agent, spec_lab, spec_cmip, end_year=2019, base_year=1750):
    """
    Scale ERF forcing in the end year (2019) by the concentrations/emissions each year
    divided by the concentration/emission in the end year (relative to the base year).
    :param forcing_tot:
    :param df_agent:
    :param spec_lab:
    :param spec_cmip:
    :param end_year:
    :param base_year:
    :return:
    """
    delta_spec_end_year = df_agent[spec_lab].loc[end_year] - df_agent[spec_lab].loc[base_year]  # 2019
    delta_spec = df_agent[spec_lab] - df_agent[spec_lab].loc[base_year]  # 1750-2019
    aerchemmip_endyear_forcing = forcing_tot[spec_cmip]  # from Bill collins
    erf_spec = aerchemmip_endyear_forcing * delta_spec / delta_spec_end_year  # scale by concentrations
    return erf_spec


# %%
fig, ax = plt.subplots(figsize=[10, 5])

ERFs = {}
for spec in ['CO2', 'N2O', 'CH4']:
    forcing_spec = scale_ERF(forcing_total_collins, df_conc, spec, spec)  # # scale by concentrations

    print(spec)
    forcing_spec.plot(label=spec)
    ERFs[spec] = forcing_spec

spec = 'NOx'
for spec in ['NOx', 'SO2', 'BC', 'OC', 'NH3']:
    forcing_spec = scale_ERF(forcing_total_collins, df_emissions, spec, spec)  # scale by emissions

    ERFs[spec] = forcing_spec

    forcing_spec.plot(label=spec)

# VOC: scale with CO emissions because these are mostly the same
spec = 'CO'

forcing_spec = scale_ERF(forcing_total_collins, df_emissions, spec, 'VOC')  # scale by concentrations

ERFs['VOC'] = forcing_spec

forcing_spec.plot(label=spec)

plt.ylabel('W m$^{-2}$')

plt.legend(loc='upper left')

# %% [markdown]
# ## HFCs:
# For HFCs we use the RE from Hodnebrog et al 2019 and the concentrations from chapter two to calculate the ERF. 

# %% [markdown]
# ### Hodnebrog et al:

# %% [markdown]
# Read in table 3 from Hodnebrog et al 

# %%
fp_hodnebrog = INPUT_DATA_DIR_BADC / 'hodnebrog_tab3.csv'
#fp_hodnebrog = INPUT_DATA_DIR / 'hodnebrog_tab3.csv'

# %%
df_hodnebrog = read_csv_badc(fp_hodnebrog, index_col=[0, 1], header=[0, 1])
df_HFC = df_hodnebrog.loc[('Hydrofluorocarbons',)]
df_HFC

# %% [markdown]
# df_hodnebrog = pd.read_csv(fp_hodnebrog, index_col=[0, 1], header=[7, 8])
# df_HFC = df_hodnebrog.loc[('Hydrofluorocarbons',)]
# df_HFC
#
# df_hodnebrog = pd.read_csv(fp_hodnebrog, index_col=[0, 1], header=[0, 1])
# df_HFC = df_hodnebrog.loc[('Hydrofluorocarbons',)]
# df_HFC

# %%
RE_df = df_HFC['RE (Wm-2ppb-1)'].transpose()
# RE_df = RE_df.reset_index().rename({'level_1':'Species'},axis=1).set_index('Species').drop('level_0', axis=1)
RE_df  # .transpose().loc['This work']*

# %%
df_conc[RE_df.columns] - df_conc[RE_df.columns].loc[1750]

# %%
ERF_HFCs = (df_conc[RE_df.columns] - df_conc[RE_df.columns].loc[1750]) * RE_df.loc['This work'] * 1e-3  # ppt to ppb
ERF_HFCs['HFCs'] = ERF_HFCs.sum(axis=1)
ERF_HFCs

# %%
for c in ERF_HFCs.columns[:-1]:
    ERF_HFCs[c].plot(label=c)
ERF_HFCs.sum(axis=1).plot(label='HFCs', color='k', linewidth=3)
plt.legend()
plt.ylabel('W/m2')

# %%
# We use CFC-12 emissions for HC
spec = 'CFC-12'
forcing_HC = scale_ERF(forcing_total_collins, df_conc, spec, 'HC')  # scale by concentrations

ERFs['HC'] = forcing_HC + ERF_HFCs['HFCs']

ERF_HFCs['HFCs']

# %%
df_ERF = pd.concat(ERFs, axis=1)  # ['CO2'].loc[1752]
df_ERF.columns

# %%
forcing_HC[2019]

# %% [markdown]
# ### Add HFC to dataset

# %%
df_collins['HFCs'] = 0
df_collins.loc['HC', 'HFCs'] = ERF_HFCs['HFCs'][2019]
df_collins

# %%
df_collins.sum(axis=0)

# %%
df_ERF

# %% [markdown]
# ### Decompose matrix:

# %%
df_coll_t = df_collins.transpose()
if 'Total' in df_coll_t.index:
    df_coll_t = df_coll_t.drop('Total')
# scale by total:
scale = df_coll_t.sum()
# normalized ERF: 
df_col_normalized = df_coll_t / scale
#
df_col_normalized.transpose().plot.barh(stacked=True)
plt.legend(bbox_to_anchor=(1, 1))

# %% [markdown]
# # Save ERFs

# %%
fn_output_ERF

# %%
fn_output_ERF.parent.mkdir(parents=True, exist_ok=True)
df_ERF.to_csv(fn_output_ERF)
df_col_normalized.to_csv(fn_output_decomposition)
df_collins.to_csv(fn_output_ERF_2019)

# %%
df_ERF
