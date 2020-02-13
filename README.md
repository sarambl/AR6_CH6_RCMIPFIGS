## Description
Contact: Sara Marie Blichner, University of Oslo [s.m.blichner@geo.uio.no](s.m.blichner@geo.uio.no)


Code for analyzing and plotting RCMIP data for AR6 IPCC. 


OBS: Some of the code is based on or copied directly with permission from [https://gitlab.com/rcmip/rcmip](https://gitlab.com/rcmip/rcmip) 
 Zebedee Nicholls ([zebedee.nicholls@climate-energy-college.org](zebedee.nicholls@climate-energy-college.org)). 


## Installation

```bash
git clone https://github.com/sarambl/AR6_CH6_RCMIPFIGS.git
cd AR6_CH6_RCMIPFIGS
conda env create -f env_rcmip_ch6.yml
conda activate rcmip_ch6
pip install -e .
``` 

## Input data: 
The input data for these figures is RCMIP phase 1 model data:
 
Nicholls, Zebedee, & Gieseke, Robert. (2019). RCMIP Phase 1 Data (Version v1.0.0)
 [Data set]. Zenodo. http://doi.org/10.5281/zenodo.3593570

For more detail on the dataset, see:

Nicholls, Z. R. J., Meinshausen, M., Lewis, J., Gieseke, R., Dommenget, D., Dorheim, K., Fan, C.-S., Fuglestvedt, J. S., Gasser, T., Golüke, U., Goodwin, P., Kriegler, E., Leach, N. J., Marchegiani, D., Quilcaille, Y., Samset, B. H., Sandstad, M., Shiklomanov, A. N., Skeie, R. B., Smith, C. J., Tanaka, K., Tsutsui, J., and Xie, Z.: Reduced complexity model intercomparison project phase 1: Protocol, results and initial observations, Geosci. Model Dev. Discuss., https://doi.org/10.5194/gmd-2019-375, in review, 2020.
 Nicholls Z. et al (2020), "Reduced complexity model intercomparison project phase 1: Protocol, results and initial observations", 
 https://www.geosci-model-dev-discuss.net/gmd-2019-375/


## Usage:  

### Get input data:
Download the data from http://doi.org/10.5281/zenodo.3593570

Create a symbolic link to the input data in [ar6_ch6_rcmipfigs/data_in](./ar6_ch6_rcmipfigs/data_in):

In project base do:
```bash
ln -s /path/to/download_data/rcmip-tmp/data ar6_ch6_rcmipfigs/data_in/
```            
  
### Preprocess data
Follow the below steps. 
0. **Create a nicely formatted dataset:**: 
Run notebook [0_database-generation.ipynb](./ar6_ch6_rcmipfigs/notebooks/0_database-generation.ipynb)
This will create the folder [data_in/database-results](./ar6_ch6_rcmipfigs/data_in/database-results) and the
data there. 
1. **Do various fixes and save relevant data as netcdf**: Run notebook 
[1_preprocess_data.ipynb](./ar6_ch6_rcmipfigs/notebooks/1_preprocess_data.ipynb)
This creates the dataset as a netcdf file in 
[ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc](ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc)
2. **Calculate delta T from effective radiative forcing:** Run notebook [2_compute_delta_T.ipynb](./ar6_ch6_rcmipfigs/notebooks/2_compute_delta_T.ipynb)
This creates at netcdf file in [ar6_ch6_rcmipfigs/data_out/dT_data_rcmip_models.nc](ar6_ch6_rcmipfigs/data_out/dT_data_rcmip_models.nc)
 [ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc](ar6_ch6_rcmipfigs/data_out/forcing_data_rcmip_models.nc)

OR: 

1. Simply run [00-02_shortcut.ipynb](./ar6_ch6_rcmipfigs/notebooks/00-02_shortcut.ipynb)

## Plot figures:
The figures are produced in notebooks:
- [3_delta_T_plot.ipynb](./ar6_ch6_rcmipfigs/notebooks/3_delta_T_plot.ipynb)
- [3-1_delta_T_plot_SLCF_sum.ipynb](./ar6_ch6_rcmipfigs/notebooks/3-1_delta_T_plot_SLCF_sum.ipynb)
- [3-2_delta_T_plot_bar_stacked.ipynb](./ar6_ch6_rcmipfigs/notebooks/3-2_delta_T_plot_bar_stacked.ipynb)
Table (sensitivity to ECS):
- [2-1_compute_delta_T_sensitivity.ipynb](./ar6_ch6_rcmipfigs/notebooks/2-1_compute_delta_T_sensitivity.ipynb)

Extra: 
- [3-2_delta_T_plot_contribution_total.ipynb](./ar6_ch6_rcmipfigs/notebooks/3-2_delta_T_plot_contribution_total.ipynb)


## Directory overview: 
 - [ar6_ch6_rcmipfigs](./ar6_ch6_rcmipfigs)
 
    - [data_in](./ar6_ch6_rcmipfigs/data_in) Input data
    - [data_out](./ar6_ch6_rcmipfigs/data_out) Output data
    - [misc](./ar6_ch6_rcmipfigs/misc) Various non-code utils
    - [notebooks](./ar6_ch6_rcmipfigs/data_out) Notebooks
    - [results](./ar6_ch6_rcmipfigs/results) Results in terms of figures and tables 
    - [utils](./ar6_ch6_rcmipfigs/utils) Code utilities  
    



## 