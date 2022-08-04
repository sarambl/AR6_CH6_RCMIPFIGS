## Description
Contact: Sara Marie Blichner, University of Oslo 

[s.m.blichner@geo.uio.no](mailto:s.m.blichner@geo.uio.no) or [sara.blichner@aces.su.no](mailto:sara.blichner@aces.su.se)


Code for analyzing and plotting for AR6 IPCC. 


Note: Thanks to Zebedee Nicholls Zebedee Nicholls ([zebedee.nicholls@climate-energy-college.org](mailto:zebedee.nicholls@climate-energy-college.org)) and Chris Smith [https://github.com/chrisroadmap](https://github.com/chrisroadmap) for supplying data and answering questions. 

Also, code in [ar6_ch6_rcmipfigs/notebooks/fig6_12_and_ts15_spm2/utils_hist_att/attribution_1750_2019_newBC_smb.py](ar6_ch6_rcmipfigs/notebooks/fig6_12_and_ts15_spm2/utils_hist_att/attribution_1750_2019_newBC_smb.py) is is only slightly modified version of code Bill Collins has written (only technical changes).
 

## RESULTS:

The resulting figures can be found in [/ar6_ch6_rcmipfigs/results](./ar6_ch6_rcmipfigs/results)



## Installation

```bash
git clone https://github.com/sarambl/AR6_CH6_RCMIPFIGS.git
cd AR6_CH6_RCMIPFIGS
conda env create -f env_rcmip_ch6.yml
conda activate rcmip_ch6
pip install -e .
cd ar6_ch6_rcmipfigs/notebooks/
python X-shortcuts.py
``` 

## Input data: 
The correct source citations will be updated soon. 

In this work we use: 
1) Impulse response function (IRF) from AR6 [ar6_ch6_rcmipfigs/data_in_badc_csv/recommended_irf_from_2xCO2_2021_02_25_222758.csv](ar6_ch6_rcmipfigs/data_in_badc_csv/recommended_irf_from_2xCO2_2021_02_25_222758.csv)
2) SSP scenario ERF from FAIR [ar6_ch6_rcmipfigs/data_in_badc_csv/SSPs/](ar6_ch6_rcmipfigs/data_in_badc_csv/SSPs)
3) ERF from Thornhill et al (2021) [ar6_ch6_rcmipfigs/data_in_badc_csv/table2_thornhill2020.csv](ar6_ch6_rcmipfigs/data_in_badc_csv/table2_thornhill2020.csv)
4) Radiative forcing for HFCs from Hodnebrog et al (2020) [ar6_ch6_rcmipfigs/data_in_badc_csv/hodnebrog_tab3.csv](ar6_ch6_rcmipfigs/data_in_badc_csv/hodnebrog_tab3.csv)
5) Historical emissions of SLCFs from CEDS from here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4509372.svg)](https://doi.org/10.5281/zenodo.4509372) [ar6_ch6_rcmipfigs/data_in/historical_delta_GSAT/CEDS_v2021-02-05_emissions](ar6_ch6_rcmipfigs/data_in/historical_delta_GSAT/CEDS_v2021-02-05_emissions)
6) Historical concentrations from AR6 [ar6_ch6_rcmipfigs/data_in/historical_delta_GSAT/LLGHG_history_AR6_v9_updated.xlsx](ar6_ch6_rcmipfigs/data_in/historical_delta_GSAT/LLGHG_history_AR6_v9_updated.xlsx)
7) Uncertainties in $\Delta$ GSAT from FAIR [ar6_ch6_rcmipfigs/data_in/slcf_warming_ranges.csv](ar6_ch6_rcmipfigs/data_in/slcf_warming_ranges.csv)




## Usage:  

  
### Preprocess data

1. Simply run [X_shortcuts.ipynb](./ar6_ch6_rcmipfigs/notebooks/X-shortcuts.ipynb)

## Plot figures:
The figures are produced in notebooks:
- [Figure 6.12 (TS15)](./ar6_ch6_rcmipfigs/notebooks/fig6_12_and_ts15_spm2/04_02_plot_fig6_12_TS15.ipynb)
- [Data for SMP fig 2](./ar6_ch6_rcmipfigs/notebooks/fig6_12_and_ts15_spm2/04_01_plot-period_fig2_SPM.ipynb)
- [Figure 6.22](./ar6_ch6_rcmipfigs/notebooks/fig6_22_and_fig6_24/03-01_plot_fig6_22_dT_lineplot.ipynb)
- - [Figure 6.24](./ar6_ch6_rcmipfigs/notebooks/fig6_22_and_fig6_24/03-02_plot_fig6_24_dT_stacked_scenario.ipynb)



## Directory overview: 
 - [ar6_ch6_rcmipfigs](./ar6_ch6_rcmipfigs)

    - [data_in](./ar6_ch6_rcmipfigs/data_in) Input data
    - [data_in_badc_csv](./ar6_ch6_rcmipfigs/data_in) Input data with added metadata. This is done for the data where it was reasonable and is used when it exists (otherwise the data in data_in is used). See notebook [notebooks/convert2badc_csv/convert2badc_csv.ipynb](ar6_ch6_rcmipfigs/notebooks/convert2badc_csv/convert2badc_csv.ipynb) for the conversion between input data and badc_csv. 
    - [data_out](./ar6_ch6_rcmipfigs/data_out) Data products produced through the notebooks, but not the final plotted data. 
    - [misc](./ar6_ch6_rcmipfigs/misc) Various non-code utils
    - [notebooks](./ar6_ch6_rcmipfigs/notebooks) Notebooks
    - [results](./ar6_ch6_rcmipfigs/results) Results in terms of figures and data plotted. 
    - [utils](./ar6_ch6_rcmipfigs/utils) Code utilities  
    

## Libraries, software etc:
A list of the required packages for these figures can be found in [env_rcmip_ch6.yml](env_rcmip_ch6.yml)

## References:

- Hodnebrog, Ø, B. Aamaas, J. S. Fuglestvedt, G. Marston, G. Myhre, C. J. Nielsen, M. Sandstad, K. P. Shine, and T. J. Wallington. “Updated Global Warming Potentials and Radiative Efficiencies of Halocarbons and Other Weak Atmospheric Absorbers.” Reviews of Geophysics 58, no. 3 (2020): e2019RG000691. https://doi.org/10.1029/2019RG000691.

- Thornhill, Gillian D., William J. Collins, Ryan J. Kramer, Dirk Olivié, Ragnhild B. Skeie, Fiona M. O’Connor, Nathan Luke Abraham, et al. “Effective Radiative Forcing from Emissions of Reactive Gases and Aerosols – a Multi-Model Comparison.” Atmospheric Chemistry and Physics 21, no. 2 (January 21, 2021): 853–74. https://doi.org/10.5194/acp-21-853-2021.


