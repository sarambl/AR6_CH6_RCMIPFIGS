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
from IPython.display import clear_output

# %%
import runpy

# %% [markdown]
# ## Preprocess:

# %%
runpy.run_path('01_preprocess.py')
clear_output()

# %% [markdown]
# ## Create $\Delta$GSAT:

# %%
print('1')
runpy.run_path('02-01_compute_delta_T_recommandation.py')
print('2***********')

runpy.run_path('02-01-02_compute_delta_T_recommandation-HFCs.py')

runpy.run_path('02-02_uncertainty.py')

runpy.run_path('02-03_uncertainty_sum_aerosols.py')
print('done')


# %% [markdown]
#
# ## Create plots etc:

# %%
runpy.run_path('03-01_delta_T_plot_recommendation.py')
runpy.run_path('03-02_delta_T_plot_bar_stacked_recommendation.py')
runpy.run_path('03-03_delta_T_plot_contribution_total_recommendation.py')
print('done')
# %% [markdown]
# ## Figures ERF emission based attribution (from Bill Collins script)

# %%
runpy.run_path('ERF_hist_attribution/make_historical_attribution.py')
print('done')

# %% [markdown]
# ## Figures ERF/GSAT emission based attribution:

# %%
runpy.run_path('GSAT_change_hist_attribution/01_Emission_based_ERFs.py')
runpy.run_path('GSAT_change_hist_attribution/02_historical_deltaGSAT.py')
runpy.run_path('GSAT_change_hist_attribution/03_plot.py')
print('done')
