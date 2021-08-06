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

# %%
import runpy

# %% [markdown]
# ## Figures ERF emission based attribution (from Bill Collins script)

# %%
runpy.run_path('GSAT_change_hist_attribution/01_make_historical_attribution.py')
print('done')

# %% [markdown]
# ## Figures historical ERF/GSAT:

# %% tags=[] jupyter={"outputs_hidden": true}
runpy.run_path('GSAT_change_hist_attribution/02_Emission_based_ERFs.py')
runpy.run_path('GSAT_change_hist_attribution/03_historical_deltaGSAT.py')
runpy.run_path('GSAT_change_hist_attribution/04_01_plot-period.py')
runpy.run_path('GSAT_change_hist_attribution/04_02_plot.py')
print('done')

# %%
'Done'

# %%
