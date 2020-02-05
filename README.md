## Description
Contact: Sara Marie Blichner, University of Oslo [s.m.blichner@geo.uio.no](s.m.blichner@geo.uio.no)


Code for analyzing and plotting RCMIP data for AR6 IPCC. 
The model data used in these figures are available here:
[https://gitlab.com/rcmip/rcmip](https://gitlab.com/rcmip/rcmip). 
For questions about this data, please contact  Zebedee Nicholls ([zebedee.nicholls@climate-energy-college.org](zebedee.nicholls@climate-energy-college.org)).

OBS: Some of the code is based on or copied directly with permission from [https://gitlab.com/rcmip/rcmip](https://gitlab.com/rcmip/rcmip) 
 Zebedee Nicholls ([zebedee.nicholls@climate-energy-college.org](zebedee.nicholls@climate-energy-college.org)). 

## Installation

```bash
git clone  git@github.com:sarambl/AR6_CH6_RCMIPFIGS.git
cd AR6_CH6_RCMIPFIGS
conda env create env_rcmip_ch6.yml
conda activate rcmip_ch6
pip install -e .
``` 
## Usage:  

### Get input data:
   - input data is [here](./ar6_ch6_rcmipfigs/data_in) 

        The input data for these plots can be found [here](https://gitlab.com/rcmip/rcmip/-/tree/master/data/)
        Download this data to some location /path/to/data and make a symbolic link in ar6_ch6_rcmipfigs.
            In project base do:
            
            ln -s /path/to/data/ ar6_ch6_rcmipfigs/data_in/
            

   - output data is [here](ar6_ch6_rcmipfigs/data_out)
  
### Postprocess data
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

### Plot figures:
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
    
    to reproduce the results and figures go to [run_me](./notebooks/run_me.ipynb)

