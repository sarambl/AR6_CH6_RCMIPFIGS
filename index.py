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
# ## Description
#
# Contact: Sara Marie Blichner, University of Oslo [s.m.blichner@geo.uio.no](s.m.blichner@geo.uio.no)
#
#
# Code for analyzing and plotting RCMIP data for AR6 IPCC. 
#
#
# OBS: Some of the code is based on or copied directly with permission from [https://gitlab.com/rcmip/rcmip](https://gitlab.com/rcmip/rcmip) 
#  Zebedee Nicholls ([zebedee.nicholls@climate-energy-college.org](zebedee.nicholls@climate-energy-college.org)). 
#

# %% [markdown]
# ## Download input data

# %%
# !wget https://zenodo.org/record/3593570/files/rcmip-phase-1-submission.tar.gz

# %%
# !cd ar6_ch6_rcmipfigs/data_in; tar zxf ../../rcmip-phase-1-submission.tar.gz; mv rcmip-tmp/data/* .; 

# %% [markdown]
# ### Preprocess data
# Follow the below steps. 
#
#
# 0. **Create a nicely formatted dataset:**: 
# Run notebook [0_database-generation.ipynb](./ar6_ch6_rcmipfigs/notebooks/0_database-generation.ipynb)
# This will create the folder [data_in/database-results](./ar6_ch6_rcmipfigs/data_in/database-results) and the
# data there. 
# 1. **Do various fixes and save relevant data as netcdf**: Run notebook 
# [1_preprocess_data.ipynb](./ar6_ch6_rcmipfigs/notebooks/1_preprocess_data.ipynb)
# This creates the dataset as a netcdf file in 
# [ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc](ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc)
# 2. **Calculate delta T from effective radiative forcing:** Run notebook [2_compute_delta_T.ipynb](./ar6_ch6_rcmipfigs/notebooks/2_compute_delta_T.ipynb)
# This creates at netcdf file in [ar6_ch6_rcmipfigs/data_out/dT_data_rcmip_models.nc](ar6_ch6_rcmipfigs/data_out/dT_data_rcmip_models.nc)
#  [ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc](ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc)
#
# OR: 
#
# 1. Simply run [X-shortcut.ipynb](./ar6_ch6_rcmipfigs/notebooks/X-shortcut.ipynb)
#
# ## Plot figures:
# The figures are produced in notebooks:
# - [3_delta_T_plot.ipynb](./ar6_ch6_rcmipfigs/notebooks/3_delta_T_plot.ipynb)
# - [3-2_delta_T_plot_bar_stacked.ipynb](./ar6_ch6_rcmipfigs/notebooks/3-2_delta_T_plot_bar_stacked.ipynb)
# Table (sensitivity to ECS):
# - [2-1_compute_delta_T_sensitivity.ipynb](./ar6_ch6_rcmipfigs/notebooks/2-1_compute_delta_T_sensitivity.ipynb)
#
# Extra: 
# - [3-2_delta_T_plot_contribution_total.ipynb](./ar6_ch6_rcmipfigs/notebooks/3-2_delta_T_plot_contribution_total.ipynb)

# %% [markdown]
# ## Directory overview: 
#  - [ar6_ch6_rcmipfigs](./ar6_ch6_rcmipfigs)
#  
#     - [data_in](./ar6_ch6_rcmipfigs/data_in) Input data
#     - [data_out](./ar6_ch6_rcmipfigs/data_out) Output data
#     - [misc](./ar6_ch6_rcmipfigs/misc) Various non-code utils
#     - [notebooks](./ar6_ch6_rcmipfigs/data_out) Notebooks
#     - [results](./ar6_ch6_rcmipfigs/results) Results in terms of figures and tables 
#     - [utils](./ar6_ch6_rcmipfigs/utils) Code utilities  
#     
#
#
#
# ## 
