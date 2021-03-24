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
# ## Make plot ERF 2019

# %%
import pandas as pd
import numpy.testing
from numpy.testing import assert_allclose
from ar6_ch6_rcmipfigs.utils.plot import get_chem_col
from ar6_ch6_rcmipfigs.constants import RESULTS_DIR, INPUT_DATA_DIR
from pathlib import  Path
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# ## Get tables from script from Bill

# %%
from ar6_ch6_rcmipfigs.notebooks.ERF_hist_attribution import attribution_1750_2019_v2_smb

# %%
table, table_sd = attribution_1750_2019_v2_smb.main(plot=True)

# %%
table.sum()#_sd

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Scale cloud forcing to fit mest estimate 0.84

# %%
table.sum()['Cloud']

# %%
(correct_cloud_forcing*table['Cloud'].sum())#.sum()

# %%
table_c = table.copy()
correct_cloud_forcing = -0.84
scale_fac = correct_cloud_forcing/table.sum()['Cloud']
table_c['Cloud']=scale_fac*table['Cloud']
table_c.sum()


# %% [markdown]
# ## Add together O3 primary and O3

# %%
o3_sum = table_c['O3']+table_c['O3_prime']
tab2 = table_c.copy(deep=True).drop(['O3','O3_prime','Total'], axis=1)
tab2['O3'] = o3_sum

# %% [markdown]
# ## Replace GHG with N2O and HC

# %%
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

# %% [markdown]
#  No need to fix std because we only use the total (which is not influenced by the summation above). 

# %%
table_sd

# %% [markdown]
# Write tables to file.

# %%
table_ed.to_csv(RESULTS_DIR/'tables_historic_attribution/table_mean_smb_orignames.csv')
table_sd.to_csv(RESULTS_DIR/'tables_historic_attribution/table_std_smb_orignames.csv')

# %%
from ar6_ch6_rcmipfigs.utils.plot import get_chem_col

# %% [markdown]
# Variables in the rigth order:

# %%
varn = ['co2','N2O','HC','ch4','o3','H2O_strat','ari','aci']
var_dir = ['CO2','N2O','HC','CH4_lifetime','O3','Strat_H2O','Aerosol','Cloud']

# %% [markdown]
# Colors:

# %%
cols = [get_chem_col(var) for var in varn]

# %% [markdown]
# ## Uncertainty:

# %% [markdown]
# We have the standard deviation, but would like the use the standard error of the mean AND we would like to calculate the 5-95th percentile. 

# %% [markdown]
# We have the standard deviation (as far as I can tell, not the unbiased one)

# %% [markdown]
# $\sigma=\sqrt {\frac {\sum _{i=1}^{n}(x_{i}-{\overline {x}})^{2}}{n}}$

# %% [markdown]
# The unbiased estimator would be:

# %% [markdown]
# $s=\sqrt {\frac {\sum _{i=1}^{n}(x_{i}-{\overline {x}})^{2}}{n-1}} = \sigma \cdot \sqrt{ \frac{n}{n-1}}$

# %% [markdown]
# The standard error is:
#
# $SE = \frac{\sigma}{n}$

# %% [markdown]
# Finally, we want 5-95th percentile. Assuming normal distribution, this amounts to multiplying the standard error by 1.645

# %%
std_2_95th = 1.645

# %%
import pandas as pd
num_mod_lab = 'Number of models (Thornhill 2020)'
thornhill = pd.read_csv(INPUT_DATA_DIR/'table2_thornhill2020.csv', index_col=0)
thornhill.index = thornhill.index.rename('Species')
thornhill

# %% [markdown]
# ![](thornhill.jpg)

# %%
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

# %% [markdown]
# ## Rename some variables

# %%
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

# %%
fn_sd = RESULTS_DIR/'tables_historic_attribution/table_uncertainties_smb_plt.csv'
fn_mean = RESULTS_DIR/'tables_historic_attribution/table_mean_smb_plt.csv'


# %%
df_err = df_err.rename(rename_dic_cols, axis=0)
df_err.to_csv(fn_sd)
tab_plt.to_csv(fn_mean)

# %%
width = 0.7
kwargs = {'linewidth':.1,'edgecolor':'k'}


# %%
import seaborn as sns

# %%
ybar = np.arange(len(tab_plt)+1)#, -1)
ybar

# %%
table_ed.sum(axis=0)

# %%
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
fn = 'attribution_1750_2019_5-95th.png'
fp = RESULTS_DIR /'figures_historic_attribution'/fn
fp.parent.mkdir(parents=True, exist_ok=True)
ax.set_yticks([])

plt.savefig(fp, dpi=300)
plt.savefig(fp.with_suffix('.pdf'), dpi=300)
plt.show()


# %%

# %%
