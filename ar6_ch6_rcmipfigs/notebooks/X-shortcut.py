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
# ## Create table on delta T dependence on ECS:

# %%
runpy.run_path('2_compute_delta_T.py')


# %% [markdown]
# ## Create plots etc:

# %%
runpy.run_path('3-1_delta_T_plot.py')

# %%
runpy.run_path('3-2_delta_T_plot_bar_stacked.py.py')
runpy.run_path('3-3_delta_T_plot_contribution_total.py')

# %%
