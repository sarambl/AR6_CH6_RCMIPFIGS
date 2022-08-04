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

# %% [markdown] pycharm={"name": "#%% md\n"} tags=[]
# ## Make plot ERF 2019
#
#
# This script uses code produced by Bill Collins to produce an emission based estimate of ERF in 2019 vs 1750 based on Thornhill et al 2021. 
#
#
# Thornhill, Gillian D., William J. Collins, Ryan J. Kramer, Dirk Olivié, Ragnhild B. Skeie, Fiona M. O’Connor, Nathan Luke Abraham, et al. “Effective Radiative Forcing from Emissions of Reactive Gases and Aerosols – a Multi-Model Comparison.” Atmospheric Chemistry and Physics 21, no. 2 (January 21, 2021): 853–74. https://doi.org/10.5194/acp-21-853-2021.
# %% pycharm={"name": "#%%\n"} tags=[]
import matplotlib.pyplot as plt
import numpy as np
# %% pycharm={"name": "#%%\n"}
import pandas as pd

from ar6_ch6_rcmipfigs.constants import RESULTS_DIR, INPUT_DATA_DIR_BADC, OUTPUT_DATA_DIR

# %% [markdown] pycharm={"name": "#%% md\n"}
# ### Output filenames.

# %% pycharm={"name": "#%%\n"}
# standard deviation filename:

fn_sd_orig_names = OUTPUT_DATA_DIR /'fig6_12_ts15_historic_delta_GSAT/table_std_thornhill_collins_orignames.csv'
# fn_sd = RESULTS_DIR / 'tables_hist_attribution_fig6_12_ts15/table_uncertainties_smb_plt.csv'
# 
# mean filename
fn_mean_orig_names = OUTPUT_DATA_DIR /'fig6_12_ts15_historic_delta_GSAT/table_mean_thornhill_collins_orignames.csv'
# fn_mean = RESULTS_DIR / 'tables_hist_attribution_fig6_12_ts15/table_mean_smb_plt.csv'


# %% pycharm={"name": "#%%\n"}
fn_sd_orig_names.parent.mkdir(parents=True, exist_ok=True)
fn_mean_orig_names.parent.mkdir(parents=True, exist_ok=True)

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Get tables from script from Bill

# %% pycharm={"name": "#%%\n"} tags=[]
from ar6_ch6_rcmipfigs.notebooks.fig6_12_and_ts15_spm2.utils_hist_att import attribution_1750_2019_newBC_smb

# %% pycharm={"name": "#%%\n"}
table, table_sd = attribution_1750_2019_newBC_smb.main(plot=True)

# %% pycharm={"name": "#%%\n"}
table.sum()#_sd

# %% pycharm={"name": "#%%\n"}

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Make one category with both CH4 from emissions and change in lifetime 

# %% pycharm={"name": "#%%\n"}
ch4_ghg_old = table.loc['CH4','GHG']
ch4_lftime_old = table.loc['CH4','CH4_lifetime']

table.loc['CH4','CH4_lifetime'] = ch4_lftime_old + ch4_ghg_old
table.loc['CH4','GHG'] = 0.

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Scale cloud forcing to fit mest estimate 0.84

# %% pycharm={"name": "#%%\n"}
table_c = table.copy()
correct_cloud_forcing = - 0.84
scale_fac = correct_cloud_forcing/table.sum()['Cloud']
table_c['Cloud']=scale_fac*table['Cloud']
table_c.sum()


# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Add together O3 primary and O3

# %% pycharm={"name": "#%%\n"}
o3_sum = table_c['O3']+table_c['O3_prime']
tab2 = table_c.copy(deep=True).drop(['O3','O3_prime','Total'], axis=1)
tab2['O3'] = o3_sum

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Replace GHG with N2O and HC

# %% pycharm={"name": "#%%\n"}
table_ed = tab2.copy(deep=True)
_ghg = tab2.loc['HC','GHG']
table_ed.loc['HC','GHG'] = 0
table_ed['HC'] = 0
table_ed.loc['HC','HC']=_ghg
table_ed
_ghg = tab2.loc['N2O','GHG']
table_ed.loc['N2O','GHG'] = 0
table_ed['N2O']=0
table_ed.loc['N2O','N2O']=_ghg
table_ed = table_ed.drop('GHG', axis=1)
table_ed

# %% [markdown] pycharm={"name": "#%% md\n"}
#  No need to fix std because we only use the total (which is not influenced by the summation above). 

# %% pycharm={"name": "#%%\n"}
table_sd

# %% [markdown] pycharm={"name": "#%% md\n"}
# Write tables to file.

# %% pycharm={"name": "#%%\n"}
table_ed.to_csv(fn_mean_orig_names)
table_sd.to_csv(fn_sd_orig_names)

# %% pycharm={"name": "#%%\n"}
from ar6_ch6_rcmipfigs.utils.plot import get_chem_col

# %% [markdown] pycharm={"name": "#%% md\n"}
# Variables in the rigth order:

# %% pycharm={"name": "#%%\n"}
varn = ['co2','N2O','HC','ch4','o3','H2O_strat','ari','aci']
var_dir = ['CO2','N2O','HC','CH4_lifetime','O3','Strat_H2O','Aerosol','Cloud']

# %% [markdown] pycharm={"name": "#%% md\n"}
# Colors:

# %% pycharm={"name": "#%%\n"}
cols = [get_chem_col(var) for var in varn]

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Uncertainty:

# %% [markdown] pycharm={"name": "#%% md\n"}
# We have the standard deviation, but would like the use the standard error of the mean AND we would like to calculate the 5-95th percentile. 

# %% [markdown] pycharm={"name": "#%% md\n"}
# We have the standard deviation (as far as I can tell, not the unbiased one)

# %% [markdown] pycharm={"name": "#%% md\n"}
# $\sigma=\sqrt {\frac {\sum _{i=1}^{n}(x_{i}-{\overline {x}})^{2}}{n}}$

# %% [markdown] pycharm={"name": "#%% md\n"}
# The unbiased estimator would be:

# %% [markdown] pycharm={"name": "#%% md\n"}
# $s=\sqrt {\frac {\sum _{i=1}^{n}(x_{i}-{\overline {x}})^{2}}{n-1}} = \sigma \cdot \sqrt{ \frac{n}{n-1}}$

# %% [markdown] pycharm={"name": "#%% md\n"}
# The standard error is:
#
# $SE = \frac{\sigma}{n}$

# %% [markdown] pycharm={"name": "#%% md\n"}
# Finally, we want 5-95th percentile. Assuming normal distribution, this amounts to multiplying the standard error by 1.645

# %% pycharm={"name": "#%%\n"}
std_2_95th = 1.645

# %% pycharm={"name": "#%%\n"}
from ar6_ch6_rcmipfigs.utils.badc_csv import read_csv_badc
num_mod_lab = 'Number of models (Thornhill 2020)'
thornhill = read_csv_badc(INPUT_DATA_DIR_BADC/'table2_thornhill2020.csv', index_col=0)
thornhill.index = thornhill.index.rename('Species')
thornhill

# %% [markdown] pycharm={"name": "#%% md\n"}
# ![](thornhill.jpg)

# %% pycharm={"name": "#%%\n"}
sd_tot = table_sd['Total_sd']
df_err= pd.DataFrame(sd_tot.rename('std'))
df_err['SE'] = df_err

df_err['SE'] = df_err['std']/np.sqrt(thornhill[num_mod_lab])
df_err['95-50_SE'] = df_err['SE']*std_2_95th
df_err.loc['CO2','95-50_SE']= df_err.loc['CO2','std']
df_err

df_err['95-50'] = df_err['std']*std_2_95th
df_err.loc['CO2','95-50']= df_err.loc['CO2','std']
df_err

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Rename some variables

# %% pycharm={"name": "#%%\n"}
rename_dic_cat = {
    'CO2':'Carbon dioxide (CO$_2$)',
    'GHG':'WMGHG',
    'CH4_lifetime': 'Methane (CH$_4$)',
    'O3': 'Ozone (O$_3$)',
    'Strat_H2O':'H$_2$O (strat)',
    'Aerosol':'Aerosol-radiation',
    'Cloud':'Aerosol-cloud',
    'N2O':'N$_2$O',
    'HC':'CFC + HCFC',

}
rename_dic_cols ={
    'CO2':'CO$_2$',
    'CH4':'CH$_4$',
    'N2O':'N$_2$O',
    'HC':'CFC + HCFC',
    'NOx':'NO$_x$',
    'VOC':'NMVOC + CO',
    'SO2':'SO$_2$',
    'OC':'Organic carbon',
    'BC':'Black carbon',
    'NH3':'Ammonia'
}
tab_plt = table_ed.loc[::-1,var_dir].rename(rename_dic_cat, axis=1).rename(rename_dic_cols, axis=0)
tab_plt

# %% pycharm={"name": "#%%\n"}
df_err = df_err.rename(rename_dic_cols, axis=0)
# df_err.to_csv(fn_sd)
# tab_plt.to_csv(fn_mean)

# %% pycharm={"name": "#%%\n"}
width = 0.7
kwargs = {'linewidth':.1,'edgecolor':'k'}


# %% pycharm={"name": "#%%\n"}
import seaborn as sns

# %% pycharm={"name": "#%%\n"}
ybar = np.arange(len(tab_plt)+1)#, -1)
ybar

# %% pycharm={"name": "#%%\n"}
table_ed.sum(axis=0)

# %% pycharm={"name": "#%%\n"}
fig, ax = plt.subplots(dpi=150)#figsize=[10,10])


tab_plt.plot.barh(stacked=True, color=cols, linewidth=.0, edgecolor='k',ax=ax, width=width)
tot = table_ed.sum(axis=1)[::-1]#table_ed['Total'][::-1]
xerr = df_err['95-50'][::-1]
y = np.arange(len(tot))
plt.errorbar(tot, y,xerr=xerr,marker='d', linestyle='None', color='k', label='Sum', )
plt.legend(frameon=False)
ax.set_ylabel('')
sns.despine()





for lab, y in zip(tab_plt.index, ybar):
        #plt.text(-1.55, ybar[i], species[i],  ha='left')#, va='left')
    plt.text(-1.9, y-0.1, lab,  ha='left')#, va='left')
plt.title('Change in effective radiative forcing from  1750 to 2019')
plt.xlabel(r'Effective radiative forcing, W m$^{-2}$')
plt.xlim(-1.5, 2.6)
    #plt.xlim(-1.6, 2.0)
sns.despine(fig, left=True, trim=True)
plt.legend(loc='lower right', frameon=False)
plt.axvline(x=0., color='k', linewidth=0.25)
#fn = 'attribution_1750_2019_5-95th.png'
#fp = RESULTS_DIR /'figures_historic_attribution'/fn
#fp.parent.mkdir(parents=True, exist_ok=True)
ax.set_yticks([])

#plt.savefig(fp, dpi=300)
#plt.savefig(fp.with_suffix('.pdf'), dpi=300)
plt.show()


# %% pycharm={"name": "#%%\n"}
