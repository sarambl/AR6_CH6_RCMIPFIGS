
## Description
Code for analyzing and plotting RCMIP data for AR6 IPCC. 

OBS: Some of the code is based on or copied directly with permission from [https://gitlab.com/rcmip/rcmip](https://gitlab.com/rcmip/rcmip) 
 Zebedee Nicholls ([zebedee.nicholls@climate-energy-college.org](zebedee.nicholls@climate-energy-college.org)). 

## installation
    ```bash
    git clone  git@github.com:sarambl/AR6_CH6_RCMIPFIGS.git
    cd AR6_CH6_RCMIPFIGS
    conda env create env_rcmip_ch6.yml
    conda activate rcmip_ch6
    pip install -e .  

## Get input data:
The input data for these plots can be found [here](https://gitlab.com/rcmip/rcmip/-/tree/master/data/)
Download this data to some location /path/to/data and make a symbolic link in ar6_ch6_rcmipfigs.
In project base do: 
    ```bash
    ln -s /path/to/data/ ar6_ch6_rcmipfigs/data_in/


## Usage:  
TODO: keep updating 
   - input data is [here](./data_in) 
        - download big_dataset from here and link to sdf 
        ```bash
      ln -s 
   - output data is [here](./data_out)
   
## Directory overview: 
    code utilities are in [here](./util)
    
    to reproduce the results and figures go to [run_me](./notebooks/run_me.ipynb)

