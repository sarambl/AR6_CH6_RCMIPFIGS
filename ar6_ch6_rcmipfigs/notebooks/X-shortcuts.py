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
# ## Runs all the scripts to produce all figures

# %%
from IPython.display import clear_output

# %%
import runpy

# %% [markdown]
# ## Figure 6.22 and 6.24:

# %% [markdown]
# ### Preprocess:

# %%
runpy.run_path('fig6_22_and_fig6_24/01_preprocess.py')
clear_output()

# %% [markdown]
# ### Create $\Delta$GSAT:

# %%
print('1')
runpy.run_path('fig6_22_and_fig6_24/02-01-01_compute_delta_T_recommandation.py')
print('2***********')

runpy.run_path('fig6_22_and_fig6_24/02-01-02_compute_delta_T_recommandation-HFCs.py')

runpy.run_path('fig6_22_and_fig6_24/02-02-01_uncertainty.py')

runpy.run_path('fig6_22_and_fig6_24/02-02-02_uncertainty_sum_aerosols.py')
print('done')


# %% [markdown] jp-MarkdownHeadingCollapsed=true
#
# ### Create plots etc:

# %% [markdown]
# #### Figure 6.22

# %%
runpy.run_path('fig6_22_and_fig6_24/03-01_plot_fig6_22_dT_lineplot.py')
print('done')
# %% [markdown]
# #### Figure 6.24

# %%
runpy.run_path('fig6_22_and_fig6_24/03-02_plot_fig6_24_dT_stacked_scenario.py')
print('done')
# %% [markdown]
# ## Fig. 6.12, TS15 and data for SMP2: Figures ERF emission based attribution (from Bill Collins script)

# %%
runpy.run_path('fig6_12_and_ts15_spm2/01_make_historical_attribution.py')
print('done')

# %% [markdown]
# ### Emission based ERFs and calculate historical GSAT:

# %% tags=[]
runpy.run_path('fig6_12_and_ts15_spm2/02_Emission_based_ERFs.py')
runpy.run_path('fig6_12_and_ts15_spm2/03_historical_deltaGSAT.py')

# %% [markdown]
# ### Plot data for figure 2 (SPM) 

# %% tags=[]
runpy.run_path('fig6_12_and_ts15_spm2/04_01_plot-period_fig2_SPM.py')

# %% [markdown]
# ### Plot data for figure 6.12 TS15

# %% tags=[]
runpy.run_path('fig6_12_and_ts15_spm2/04_02_plot_fig6_12_TS15.py')
print('done')

# %%
print('Done')
