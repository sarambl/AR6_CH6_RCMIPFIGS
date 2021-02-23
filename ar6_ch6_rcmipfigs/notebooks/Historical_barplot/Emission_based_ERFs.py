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
# ## Derive emission based historical ERFs:
# We use timeseries for historical emissions/concentrations and then scale these with the 1750-2019 ERF from Thornhill (2019)/bill collins plot to derive ERF timeseries. 
# For short lived components, we use change in emissions from CEDS:
# - NOx, VOC/CO, SO2, OC, BC, NH3
# For longer lived components, we use change in concentrations from chap 2:
# - CO2, CH4, N2O, HC
#
# For ERF from VOC/CO we use emissions of CO to scale. 
# For HC we use the HC from Thornhill (which includes only CFCs and HCFCs) and additionally, we use HFC concentrations (ch2) and multipy with radiative efficiencies (RE) from Hognebrog et al (2019). 
# HC is then the sum of the HC from Thornhill (scaled with concentrations) and the HFC ERF.
#
# Finally, these ERFs are integrated with the IRF and the change in GSAT is calculated. 

# %% [markdown]
# ## idea:

# %% [markdown]
# Use estimated ERF from xx (BILL COLLINS, plot below) together with change in emissions or concentrations of gases to estimate a historical timeseries for ERF. From this, use an impulse response function (from rcmip?) to estimate the change in GSAT from each component. 

# %% [markdown]
# ![](../../results/figures_historic_attribution/attribution_1750_2019.png)

# %% [markdown]
# ## From 1750 to 2019

# %% [markdown]
# ### Input data:
# "We would have to take the present day estimates and construct historic ERF-timeseries for each component by simple scaling with global historical emissions, i.e. ignoring any non-linearities."
# - Splits from Bills figure? http://www.globalchange.umd.edu/ceds/

# %% [markdown]
# Summary Chris email:
# - CEDS for BC, OC, SO2, NH3, NOx, VOC, NH3 from here: https://zenodo.org/record/4025316#.YDAg9tj7TIU
# - The biomass burning component of emissions I use from van Marle, the CMIP6 dataset. Where I have to choose a representative scenario for the last few years I just pick SSP2-4.5. There's a lot less variation in the biomass burning scenarios than there are for the anthropogenic.
# - 

# %% [markdown]
# **concentrations**: 
# Chris sendt file? LLGHG_history_AR6_v9_updated.xlsx
#
# rcmip https://www.google.com/url?q=https%3A%2F%2Frcmip-protocols-au.s3-ap-southeast-2.amazonaws.com%2Fv5.1.0%2Frcmip-concentrations-annual-means-v5-1-0.csv&sa=D&sntz=1&usg=AFQjCNFKdZQQeBdz_W1XfaSso8aStOr0xQ
#

# %% [markdown]
# **emissions**: 
# CEDS gives up to 2019 from https://zenodo.org/record/4025316#.YDJl3XVKg5l
# - BC
# - CO
# - NH3
# - NMVOC
# - NOx
# - 

# %% [markdown]
# ## TODO:
# - load concentrations in a good format
# - load emissions in a good format
# - load bill collins 
#

# %%
import numpy as np
import pandas as pd


# %%
from ar6_ch6_rcmipfigs.constants import INPUT_DATA_DIR, BASE_DIR, OUTPUT_DATA_DIR

# %%
fn_concentrations = '../../data_in/historical_delta_GSAT/LLGHG_history_AR6_v9_updated.xlsx'

# %%
import pandas as pd

# %% [markdown]
# ## Load concentration file and interpolate from 1750 to 1850

# %%
df_conc = pd.read_excel(fn_concentrations, header=22, index_col=0)
# For C8F18 there appears to be an error in the spreadsheet where 2015 is entered as zero, presumably 0.09 but treat as missing
df_conc.loc[2015, 'C8F18'] = np.nan


# datetime index
df_conc.index = pd.to_datetime(df_conc.index, format='%Y')

# resample to yearly, i.e. NaNs will be filled between 1750 and 1850:
df_conc = df_conc.resample('Y').first()#.interpolate()
# Interpolate:
df_conc = df_conc.interpolate(method='linear',limit_area = 'inside') # 'inside' only fill values with valid on both sides. 
# reset index to year
df_conc.index = df_conc.index.year
df_conc

# %%
[c for c in df_conc.columns if 'CFC' in c]

# %% [markdown]
# ## Emissions:

# %%
import glob
from pathlib import Path


# %%
path_emissions = Path('../../data_in/historical_delta_GSAT/CEDS_v2021-02-05_emissions/')
fl =list(path_emissions.glob('*global_CEDS_emissions_by_sector_2021_02_05.csv'))

# %%
list_df_em=[]
units_dic = {}
for fn in fl:
    _df = pd.read_csv(fn)
    u_em = _df['em'].unique()
    if len(u_em)>1:
        print('double check, emission labels :')
        print(u_em)
    _em = u_em[0]
    u_units = _df['units'].unique()
    if len(u_units)>1:
        print('double check, units labels :')
        print(u_units)
    _un = u_units[0]
    _df_s = _df.drop(['em','sector','units'], axis=1).sum()
    _df_s.index = pd.to_datetime(_df_s.index, format='X%Y').year
    _df = pd.DataFrame(_df_s, columns=[_em])
    units_dic[_em]=_un
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
fp_collins = BASE_DIR/'notebooks/bill_collins/table_mean_smb_orignames.csv'

# %%
df_collins = pd.read_csv(fp_collins, index_col=0)
df_collins.index = df_collins.index.rename('emission_experiment')

# %%
df_collins#.columns

# %%
forcing_total_collins = df_collins['Total']

# %%
import matplotlib.pyplot as plt


# %%
def scale_ERF(forcing_tot, df_agent, spec, spec_cmip, end_year=2019, base_year=1750):
    delta_spec_end_year = df_agent[spec].loc[end_year]- df_agent[spec].loc[base_year] # co2 concentration in 2019
    delta_spec = df_agent[spec] -df_agent[spec].loc[base_year]# co2 concentration 1750-2019
    aerchemmip_endyear_forcing = forcing_tot[spec_cmip] # from Bill collins
    forcing_spec = aerchemmip_endyear_forcing*delta_spec/delta_spec_end_year # scale by concentrations
    return forcing_spec


# %%
fig, ax = plt.subplots(figsize=[10,5])

ERFs ={}
for spec in ['CO2','N2O','CH4']:
    forcing_spec = scale_ERF(forcing_total_collins, df_conc,spec, spec)# # scale by concentrations

    print(spec)
    forcing_spec.plot(label=spec)
    ERFs[spec]=forcing_spec





spec = 'NOx'
for spec in ['NOx','SO2','BC','OC','NH3']:
    forcing_spec = scale_ERF(forcing_total_collins, df_emissions,spec, spec)# scale by emissions

    ERFs[spec]=forcing_spec

    forcing_spec.plot(label=spec)
    
    
# VOC: scale with CO emissions because these are mostly the same
spec = 'CO'

forcing_spec = scale_ERF(forcing_total_collins, df_emissions,spec, 'VOC')#aerchemmip_2019_forcing*delta_spec_conc/delta_spec_conc2019 # scale by concentrations

ERFs['VOC']=forcing_spec

forcing_spec.plot(label=spec)


plt.ylabel('W m$^{-2}$')

plt.legend(loc='upper left')

# %%
ERFs['CO2'].index

# %%
ERFs['NOx'].index

# %%






# %% [markdown]
# # HFCs:
# For HFCs we use the RE from Hodnebrog et al 2019 and the concentrations from chapter two to calculate the ERF. 

# %% [markdown]
# ## Hodnebrog:

# %% [markdown]
# Read in table 3 from Hodnebrog et al 

# %%
fp_hodnebrog = INPUT_DATA_DIR/'hodnebrog_tab3.csv'

# %%
df_hodnebrog = pd.read_csv(fp_hodnebrog, index_col=[0,1],header=[0,1])
df_HFC =  df_hodnebrog.loc[('Hydrofluorocarbons',)]
df_HFC

# %%
RE_df = df_HFC['RE (Wm-2ppb-1)'].transpose()
#RE_df = RE_df.reset_index().rename({'level_1':'Species'},axis=1).set_index('Species').drop('level_0', axis=1)
RE_df#.transpose().loc['This work']*

# %%
df_conc[RE_df.columns] - df_conc[RE_df.columns].loc[1750] 

# %%
ERF_HFCs = (df_conc[RE_df.columns] -  - df_conc[RE_df.columns].loc[1750])*RE_df.loc['This work']*1e-3 #ppt to ppb
ERF_HFCs['HFCs'] = ERF_HFCs.sum(axis=1)
ERF_HFCs

# %%
for c in ERF_HFCs.columns[:-1]:
    ERF_HFCs[c].plot(label=c)
ERF_HFCs.sum(axis=1).plot(label='HFCs',color='k',linewidth=3)
plt.legend()
plt.ylabel('W/m2')

# %%
# We use CFC-12 emissions for HC
spec = 'CFC-12'
forcing_HC = scale_ERF(forcing_total_collins, df_conc,spec, 'HC') # scale by concentrations



ERFs['HC'] = forcing_HC + ERF_HFCs['HFCs']
ERFs['HC']

# %%
df_ERF = pd.concat(ERFs, axis=1)#['CO2'].loc[1752]
df_ERF.columns

# %%
fn = OUTPUT_DATA_DIR/'historic_delta_GSAT/hist_ERF_est.csv'
fn.parent.mkdir(parents=True,exist_ok=True)
df_ERF.to_csv(fn)
