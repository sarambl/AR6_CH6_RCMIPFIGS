## Description
Contact: Sara Marie Blichner, University of Oslo [s.m.blichner@geo.uio.no](s.m.blichner@geo.uio.no)


Code for analyzing and plotting RCMIP data for AR6 IPCC. 


Note: Thanks to Zebedee Nicholls Zebedee Nicholls ([zebedee.nicholls@climate-energy-college.org](zebedee.nicholls@climate-energy-college.org)) and Chris Smith [https://github.com/chrisroadmap](https://github.com/chrisroadmap) for supplying data and answering questions.  
 

## RESULTS:

The resulting figures can be found in [/ar6_ch6_rcmipfigs/results](./ar6_ch6_rcmipfigs/results)



## Installation

```bash
git clone https://github.com/sarambl/AR6_CH6_RCMIPFIGS.git
cd AR6_CH6_RCMIPFIGS
conda env create -f env_rcmip_ch6.yml
conda activate rcmip_ch6
pip install -e .
``` 

## Input data: 
The correct source citations will be updated soon. 

In this work we use: 
1) Impulse response function (IRF) from AR6, ?? 
2) SSP scenario ERF from FAIR
3) ERF from Thornhill et al (2021)
4) Radiative forcing for HFCs from Hodnebrog et al (2020)
5) Historical emissions of SLCFs from CEDS
6) Historical concentrations from AR6
7) Uncertainties in $\Delta$ GSAT from FAIR




## Usage:  

### Get input data:
Download the data from http://doi.org/10.5281/zenodo.3593570
E.g. do:
```bash
wget https://zenodo.org/record/3593570/files/rcmip-phase-1-submission.tar.gz
```
Unpack the data in [ar6_ch6_rcmipfigs/data_in](./ar6_ch6_rcmipfigs/data_in):
```bash 
cd ar6_ch6_rcmipfigs/data_in; tar zxf ../../rcmip-phase-1-submission.tar.gz; mv rcmip-tmp/data/* .;
```
OR: Create a symbolic link to the downloaded input data in [ar6_ch6_rcmipfigs/data_in](./ar6_ch6_rcmipfigs/data_in).
In project base do:
```bash
ln -s /path/to/download_data/rcmip-tmp/data ar6_ch6_rcmipfigs/data_in/
```            
  
### Preprocess data

1. Simply run [X_shortcuts.ipynb](./ar6_ch6_rcmipfigs/notebooks/00-02_shortcut.ipynb)

## Plot figures:
The figures are produced in notebooks:
- [03-01_delta_T_plot_recommendation.ipynb](./ar6_ch6_rcmipfigs/notebooks/03-01_delta_T_plot_recommendation.ipynb)
- [03-02_delta_T_plot_bar_stacked_recommendation.ipynb](./ar6_ch6_rcmipfigs/notebooks/03-02_delta_T_plot_bar_stacked_recommendation.ipynb)
- [03-03_delta_T_plot_contribution_total_recommendation.ipynb](./ar6_ch6_rcmipfigs/notebooks/03-03_delta_T_plot_contribution_total_recommendation.ipynb)
Table (save values to file):
- [04-01-Table_2040_2100.ipynb](./ar6_ch6_rcmipfigs/notebooks/04-01-Table_2040_2100.ipynb)
- [04-02-Table_all_years.ipynb](./ar6_ch6_rcmipfigs/notebooks/04-02-Table_all_years.ipynb)

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
    

## Libraries, software etc:
A list of the required packages for these figures can be found in [env_rcmip_ch6.yml](env_rcmip_ch6.yml)

## References:

- Hodnebrog, Ø, B. Aamaas, J. S. Fuglestvedt, G. Marston, G. Myhre, C. J. Nielsen, M. Sandstad, K. P. Shine, and T. J. Wallington. “Updated Global Warming Potentials and Radiative Efficiencies of Halocarbons and Other Weak Atmospheric Absorbers.” Reviews of Geophysics 58, no. 3 (2020): e2019RG000691. https://doi.org/10.1029/2019RG000691.

- Thornhill, Gillian D., William J. Collins, Ryan J. Kramer, Dirk Olivié, Ragnhild B. Skeie, Fiona M. O’Connor, Nathan Luke Abraham, et al. “Effective Radiative Forcing from Emissions of Reactive Gases and Aerosols – a Multi-Model Comparison.” Atmospheric Chemistry and Physics 21, no. 2 (January 21, 2021): 853–74. https://doi.org/10.5194/acp-21-853-2021.


