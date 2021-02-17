# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:18:00 2019

@author: wcoll
"""
# use pandas instead
import pandas as pd
import numpy.testing
from numpy.testing import assert_allclose
import numpy as np
import matplotlib.pyplot as plt
from ar6_ch6_rcmipfigs.utils.plot import get_chem_col

from ar6_ch6_rcmipfigs.notebooks.bill_collins.co2_forcing_AR6 import co2_forcing_AR6
from ar6_ch6_rcmipfigs.notebooks.bill_collins.ch4_forcing_AR6 import ch4_forcing_AR6
from ar6_ch6_rcmipfigs.notebooks.bill_collins.n2o_forcing_AR6 import n2o_forcing_AR6
import seaborn as sns
#%load_ext autoreload
#%load_ext autoreload
#import os
#os.chdir('ar6_ch6_rcmipfigs/notebooks/bill_collins')
# %%
def main():
    # %%
    co2_1850 = 286.7 # LLGHG_history_AR6_v8a
    co2_2014 = 397.12
    ch4_1850 = 807.6 # LLGHG_history_AR6_v8a
    ch4_2014 = 1822.88 #

    n2o_1850 = 272.5 # LLGHG_history_AR6_v8a
    n2o_2014 = 327.37

    #Rapid adjustments to WMGHGs
    co2_ra = 0.05 # FGD
    ch4_ra = -0.14 # FGD
    n2o_ra = 0.07 # FGD


    tot_em_co2 = 582. # Cumulative C since 1850 - from MAGICC input files

    ch4_erf = ch4_forcing_AR6(ch4_2014, ch4_1850, n2o_1850)*(1+ch4_ra)
    n2o_erf = n2o_forcing_AR6(n2o_2014, n2o_1850, co2_1850, ch4_1850)*(1+n2o_ra)
    hc_erf = 0.397 # 1850-2011 FGD

    erf_bc = 0.15 # Thornhill et al.
    irf_ari = -0.3 # AR6 FGD 1850-2014
    erf_aci = -1.0 # AR6 FGD 1850-2014

    inp_df, inp_sd_df,ac, ac_sd, ari, ari_sd, data, erf, erf_sd, lifech4, lifech4_sd, nspec, rfo3, rfo3_sd = getset_input_data()
# %%
    i_ch4 = np.where(data['Experiment']=='CH4')[0][0]
    i_nox = np.where(data['Experiment']=='NOx')[0][0]
    i_voc = np.where(data['Experiment']=='VOC')[0][0]
    i_n2o = np.where(data['Experiment']=='N2O')[0][0]
    i_hc = np.where(data['Experiment']=='HC')[0][0]
    i_gas = np.array([i_ch4, i_n2o, i_hc, i_nox, i_voc])
    i_non_ch4 = np.array([i_n2o, i_hc, i_nox, i_voc])

    # smb:
    # total_o3 = np.sum(rfo3)
    labs_gas = ['CH4','N2O','HC','NOx','VOC']
    labs_non_ch4 = ['N2O','HC','NOx','VOC']


    alpha = 1.30 # From chapter 6

    #print(alpha)
    ch4, ch4_sd = getset_ch4(alpha, ch4_2014, i_ch4, i_non_ch4, labs_non_ch4, lifech4, lifech4_sd)
    inp_sd_df['ch4_sd'] =ch4_sd
    inp_df['ch4'] = ch4

    #ch4_sd[i_ch4] = (ch4[i_ch4]-ch4_2014)* \
    #                np.sqrt(np.sum(np.square(lifech4_sd[[i_non_ch4]])))/ \
    #                np.sum(lifech4[i_non_ch4])

    # Ozone primary mode
    rfo3_prime, rfo3_prime_sd = get_ozone_primary_mode(ch4, ch4_1850, ch4_2014, i_ch4, rfo3, rfo3_sd)
    # %%
    # CH4 forcing
    rfch4, rfch4_sd = get_ch4_forcing(ch4, ch4_2014, ch4_ra, ch4_sd, inp_df, inp_sd_df, n2o_2014, nspec)



    # CO2:
    _em_co2 = np.zeros(nspec)
    em_co2 = pd.Series(np.zeros(nspec), index=inp_df.index)
    # todo: where do these values come from?
    em_co2['CH4'] = 6.6
    em_co2['HC'] = 0.02
    em_co2['VOC']= 26.
    _em_co2[[i_ch4, i_hc, i_voc]] = [6.6, 0.02, 26.]
    numpy.testing.assert_array_equal(em_co2, _em_co2)
    # %%
    # From MAGICC input files
    #  CH4 HC VOC, CO CO2 scalings applied of 75%, 100%, 50%, 100%
    # Assume 88% of CH4 emitted oxidised (12% remains as CH4)

    co2, rfco2, rfco2_co2 = get_co2_rf(co2_1850, co2_2014, co2_ra, em_co2, inp_df, n2o_2014, nspec, tot_em_co2)
    # %%
    #Set up WMGHG direct ERFs
    _rfghg = np.zeros(nspec)
    rfghg = pd.Series(np.zeros(nspec), index=inp_df.index)
    _rfghg[i_ch4] = ch4_erf
    _rfghg[i_n2o] = n2o_erf
    _rfghg[i_hc] = hc_erf
    rfghg['CH4'] = ch4_erf
    rfghg['N2O'] = n2o_erf
    rfghg['HC'] = hc_erf
    numpy.testing.assert_array_equal(rfghg, _rfghg)
    rfghg_sd = rfghg*0.14 # assume 14% for all WMGHGs
    #Aerosols
    #Set indicies
    i_bc = np.where(data['Experiment']=='BC')[0][0]
    i_oc = np.where(data['Experiment']=='OC')[0][0]
    i_so2 = np.where(data['Experiment']=='SO2')[0][0]
    i_nh3 = np.where(data['Experiment']=='NH3')[0][0]
    i_aer = np.array([i_bc, i_oc, i_so2, i_nh3]) # all aerosols
    i_scat = np.array([i_oc, i_so2, i_nh3]) # scattering aerosols
    labs_aer = ['BC','OC','SO2','NH3']
    labs_scat =['OC','SO2','NH3']
    list(inp_df.iloc[i_aer].index)
    inp_df.iloc[i_scat].index


    #Set aerosol ari to be erf-ac to ensure components add to erf

    #ari[i_aer] = erf[i_aer]-ac[i_aer]
    #ari_sd[i_aer] = np.sqrt(erf_sd[i_aer]**2 +ac_sd[i_aer]**2)
    ari[labs_aer] = erf[labs_aer]-ac[labs_aer]
    ari_sd[labs_aer]  = np.sqrt(erf_sd[labs_aer]**2 +ac_sd[labs_aer]**2)

    # scale SO2+OC to get total ari
    # %%
    _irf_ari_scat = irf_ari-ari[i_bc] # Set non-BC ari to 7.3.3 FGD
    irf_ari_scat = irf_ari-ari['BC'] # Set non-BC ari to 7.3.3 FGD
    #ari_scat = np.sum(ari[i_scat])
    ari_scat = np.sum(ari[labs_scat])
    #ari[i_scat] = ari[i_scat]*irf_ari_scat/ari_scat
    ari[labs_scat] = ari[labs_scat]*irf_ari_scat/ari_scat
    ari_sd[labs_scat] = ari_sd[labs_scat]*irf_ari_scat/ari_scat
    # %%
    # scale aci to get total aci from 7.3.3
    #total_aci = np.sum(ac[i_aer])
    total_aci = np.sum(ac[labs_aer])
    #ac[i_aer] = ac[i_aer]*erf_aci/total_aci
    ac[labs_aer] = ac[labs_aer]*erf_aci/total_aci
    #ac_sd[i_aer] = ac_sd[i_aer]*erf_aci/total_aci
    ac_sd[labs_aer] = ac_sd[labs_aer]*erf_aci/total_aci
    # %%
    categories = [ 'CO2', 'GHG', 'CH4_lifetime', 'O3',
                   'O3_prime', 'Strat_H2O', 'Aerosol', 'Cloud', 'Total']
    categories = [ 'CO2', 'GHG', 'CH4_lifetime', 'O3',
                   'O3_prime', 'Strat_H2O', 'Aerosol', 'Cloud', 'Total']

    df_out = pd.DataFrame(np.zeros([nspec,nspec]), columns=categories,  index=inp_df.index)
    df_out.loc[:,:] = pd.NA
    df_out_sd = pd.DataFrame(np.zeros([nspec,nspec]), columns=categories,  index=inp_df.index)
    df_out_sd.loc[:,:] = pd.NA

    df_out
    # %%
    table = np.zeros(nspec+1,
                         dtype={'names':
                                ['Species', 'CO2', 'GHG', 'CH4_lifetime', 'O3',
                                 'O3_prime', 'Strat_H2O', 'Aerosol', 'Cloud', 'Total'],
                                 'formats':
                                     ['U20', 'f8', 'f8', 'f8', 'f8',
                                      'f8', 'f8', 'f8', 'f8', 'f8']})
    table_sd = np.zeros(nspec+1,
                         dtype={'names':
                                ['Species', 'CO2_sd', 'GHG_sd', 'CH4_lifetime_sd',
                                 'O3_sd', 'O3_prime_sd', 'Strat_H2O_sd',
                                 'Aerosol_sd', 'Cloud_sd', 'Total_sd'],
                                 'formats':
                                     ['U20', 'f8', 'f8', 'f8', 'f8',
                                      'f8', 'f8', 'f8', 'f8', 'f8']})
    # %%
    df_out.loc['CO2','CO2'] = rfco2_co2
    df_out.loc['CO2','Total'] = rfco2_co2
    df_out_sd.loc['CO2','CO2'] = rfco2_co2*0.12
    df_out_sd.loc['CO2','Total'] = rfco2_co2*0.12
    # %%

    table['Species'][0] = 'CO2'
    table['CO2'][0] = rfco2_co2
    table['Total'][0] = rfco2_co2
    table_sd['Species'][0] = 'CO2'
    table_sd['CO2_sd'][0] = rfco2_co2*0.12 # 12% uncertainty
    table_sd['Total_sd'][0] = rfco2_co2*0.12
    # %%
    for ispec in np.arange(nspec):
        table['Species'][ispec+1] = data['Experiment'][ispec]
        table['CO2'][ispec+1] = rfco2[ispec]
        table['GHG'][ispec+1] = rfghg[ispec]
        table['CH4_lifetime'][ispec+1] = rfch4[ispec]
        table['O3'][ispec+1] = rfo3[ispec]
        table['O3_prime'][ispec+1] = rfo3_prime[ispec]
        table['Aerosol'][ispec+1] = ari[ispec]
        table['Cloud'][ispec+1] = ac[ispec]
        table['Total'][ispec+1] = np.sum([rfco2[ispec], rfghg[ispec], rfch4[ispec],
             rfo3[ispec], rfo3_prime[ispec], ari[ispec], ac[ispec]])
        table_sd['Species'][ispec+1] = data['Experiment'][ispec]
        table_sd['CO2_sd'][ispec+1] = rfco2[ispec]*0.12
        table_sd['GHG_sd'][ispec+1] = rfghg_sd[ispec]
        table_sd['CH4_lifetime_sd'][ispec+1] = rfch4_sd[ispec]
        table_sd['O3_sd'][ispec+1] = rfo3_sd[ispec]
        table_sd['O3_prime_sd'][ispec+1] = rfo3_prime_sd[ispec]
        table_sd['Aerosol_sd'][ispec+1] = ari_sd[ispec]
        table_sd['Cloud_sd'][ispec+1] = ac_sd[ispec]
        table_sd['Total_sd'][ispec+1] = np.sqrt(np.sum(np.square(
                [rfco2[ispec]*0.12, rfghg_sd[ispec], rfch4_sd[ispec],
                 rfo3_sd[ispec]+rfo3_prime_sd[ispec], ari_sd[ispec], ac_sd[ispec]])))
    table['Strat_H2O'][i_ch4+1] = 0.05
    table['Total'][i_ch4+1] += 0.05
    table_sd['Strat_H2O_sd'][i_ch4+1] = 0.05
    table_sd['Total_sd'][i_ch4+1] = np.sqrt(np.sum(np.square(
            [rfco2[i_ch4]*0.12, rfghg_sd[i_ch4]+rfch4_sd[i_ch4],
                 rfo3_sd[i_ch4]+rfo3_prime_sd[i_ch4], 0.05,
                 ari_sd[i_ch4], ac_sd[i_ch4]])))

    # %%
    for spec in inp_df.index:
        df_out.loc[spec]
        #table['Species'][ispec+1] = data['Experiment'][ispec]
        df_out.loc[spec,'CO2'] = rfco2[spec]
        df_out.loc[spec,'GHG'] = rfghg[spec]
        df_out.loc[spec,'CH4_lifetime'] = rfch4[spec]
        df_out.loc[spec,'O3'] = rfo3[spec]
        df_out.loc[spec,'O3_prime'] = rfo3_prime[spec]
        df_out.loc[spec,'Aerosol'] = ari[spec]
        df_out.loc[spec,'Cloud'] = ac[spec]
        df_out.loc[spec,'Total'] = np.sum([rfco2[spec], rfghg[spec], rfch4[spec],
                                           rfo3[spec], rfo3_prime[spec], ari[spec],
                                           ac[spec]])

        df_out_sd.loc[spec,'CO2'] = rfco2[spec]*0.12
        df_out_sd.loc[spec,'GHG'] = rfghg_sd[spec]
        df_out_sd.loc[spec,'CH4_lifetime'] = rfch4_sd[spec]
        df_out_sd.loc[spec,'O3'] = rfo3_sd[spec]
        df_out_sd.loc[spec,'O3_prime'] = rfo3_prime_sd[spec]
        df_out_sd.loc[spec,'Aerosol'] = ari_sd[spec]
        df_out_sd.loc[spec,'Cloud'] = ac_sd[spec]
        df_out_sd.loc[spec,'Total'] = np.sqrt(np.sum(np.square(
            [rfco2[spec]*0.12, rfghg_sd[spec], rfch4_sd[spec],
             rfo3_sd[spec]+rfo3_prime_sd[spec], ari_sd[spec], ac_sd[spec]])))

        #table_sd['Species'][ispec+1] = data['Experiment'][ispec]
        #table_sd['CO2_sd'][ispec+1] = rfco2[ispec]*0.12
        #table_sd['GHG_sd'][ispec+1] = rfghg_sd[ispec]
        #table_sd['CH4_lifetime_sd'][ispec+1] = rfch4_sd[ispec]
        #table_sd['O3_sd'][ispec+1] = rfo3_sd[ispec]
        #table_sd['O3_prime_sd'][ispec+1] = rfo3_prime_sd[ispec]
        #table_sd['Aerosol_sd'][ispec+1] = ari_sd[ispec]
        #table_sd['Cloud_sd'][ispec+1] = ac_sd[ispec]
        #table_sd['Total_sd'][ispec+1] = np.sqrt(np.sum(np.square(
        #    [rfco2[ispec]*0.12, rfghg_sd[ispec], rfch4_sd[ispec],
        #     rfo3_sd[ispec]+rfo3_prime_sd[ispec], ari_sd[ispec], ac_sd[ispec]])))

    # TODO: where does this come from
    df_out.loc['CH4','Strat_H2O'] = 0.05
    df_out.loc['CH4','Total'] += 0.05
    df_out_sd.loc['CH4','Strat_H2O'] = 0.05
    l_ch4 = 'CH4'
    df_out_sd.loc['CH4','Total'] = np.sqrt(np.sum(np.square(
        [rfco2[l_ch4]*0.12, rfghg_sd[l_ch4]+rfch4_sd[l_ch4],
         rfo3_sd[l_ch4]+rfo3_prime_sd[l_ch4], 0.05,
         ari_sd[l_ch4], ac_sd[l_ch4]])))


    table['Strat_H2O'][i_ch4+1] = 0.05
    table['Total'][i_ch4+1] += 0.05
    table_sd['Strat_H2O_sd'][i_ch4+1] = 0.05
    table_sd['Total_sd'][i_ch4+1] = np.sqrt(np.sum(np.square(
        [rfco2[i_ch4]*0.12, rfghg_sd[i_ch4]+rfch4_sd[i_ch4],
         rfo3_sd[i_ch4]+rfo3_prime_sd[i_ch4], 0.05,
         ari_sd[i_ch4], ac_sd[i_ch4]])))
    # %%
    df_out.to_csv('attribution_output_smb.csv')
    df_out_sd.to_csv('attribution_output_sd_smb.csv')

    # %%
    np.savetxt("attribution_output.csv", table, delimiter=',',
               fmt='%15s'+9*', %8.3f',
               header=','.join(table.dtype.names))
    np.savetxt("attribution_output_sd.csv", table_sd, delimiter=',',
               fmt='%15s'+9*', %8.3f',
               header=','.join(table_sd.dtype.names))


    # %%
    df_out.drop('Total',axis=1).plot.barh(stacked=True)
    plt.show()
    # %%
    fig = plt.figure()
    width = 0.7
    species =[r'CO$_2$', r'CH$_4$', r'N$_2$O', 'Halocarbon', r'NO$_X$', 'VOC', r'SO$_2$',
              'Organic Carbon', 'Black Carbon', 'Ammonia']
    #exp_list = \
    #    np.array([i_ch4, i_n2o, i_hc, i_nox, i_voc, i_so2, i_oc, i_bc, i_nh3])
    ybar = np.arange(nspec+1, 0, -1)
    labels = [r'CO$_2$', 'WMGHG',  r'CH$_4$ lifetime', r'Ozone (O$_3$)', 'Aerosol (ari)', 'Aerosol (aci)']

    pos_ghg = np.zeros(nspec+1)
    pos_ch4 = np.zeros(nspec+1)
    pos_o3 = np.zeros(nspec+1)
    pos_aer = np.zeros(nspec+1)
    pos_cloud = np.zeros(nspec+1)
    pos_h2o = np.zeros(nspec+1)
    pos_co2 = np.zeros(nspec+1)
    neg_ch4 = np.zeros(nspec+1)
    neg_o3 = np.zeros(nspec+1)
    neg_aer = np.zeros(nspec+1)
    neg_cloud = np.zeros(nspec+1)

    #CO2
    pos_co2[0] =rfco2_co2 ; pos_ghg[0] = pos_co2[0] ; pos_ch4[0] = pos_co2[0]
    pos_o3[0]=pos_co2[0]; pos_h2o[0] = pos_co2[0]
    pos_aer[0] = pos_co2[0]; pos_cloud[0] = pos_co2[0]
    print(pos_ghg)
    # Gases
    pos_co2[i_gas+1] = rfco2[i_gas]
    pos_ghg[i_gas+1] = pos_co2[i_gas+1]+rfghg[i_gas]
    print(pos_ghg)
    pos_ch4[i_gas+1] = pos_ghg[i_gas+1]+\
        np.maximum(rfch4[i_gas], 0.)
    neg_ch4[i_gas+1] = np.minimum(rfch4[i_gas], 0.)
    pos_o3[i_gas+1] = pos_ch4[i_gas+1]+\
        np.maximum(rfo3[i_gas]+rfo3_prime[i_gas], 0.)
    neg_o3[i_gas+1] = neg_ch4[i_gas+1]+\
        np.minimum(rfo3[i_gas]+rfo3_prime[i_gas], 0.)
    pos_h2o[i_gas+1] = pos_o3[i_gas+1]
    pos_h2o[i_ch4+1] += 0.05 # AR6 FGD
    pos_aer[i_gas+1] = pos_h2o[i_gas+1]+\
        np.maximum(ari[i_gas], 0.)
    neg_aer[i_gas+1] = neg_o3[i_gas+1]+\
        np.minimum(ari[i_gas], 0.)
    pos_cloud[i_gas+1] = pos_aer[i_gas+1]+\
        np.maximum(ac[i_gas], 0.)
    neg_cloud[i_gas+1] = neg_aer[i_gas+1]+\
        np.minimum(ac[i_gas], 0.)

    #Aerosols
    pos_aer[i_aer+1] = np.maximum(ari[i_aer], 0.)
    neg_aer[i_aer+1] = np.minimum(ari[i_aer], 0.)
    pos_cloud[i_aer+1] = pos_aer[i_aer+1]+\
        np.maximum(ac[i_aer], 0.)
    neg_cloud[i_aer+1] = neg_aer[i_aer+1]+\
        np.minimum(ac[i_aer], 0.)


    error = np.zeros(nspec+1)
    error[0] = co2[0]*0.12 # 12% uncertainty
    error[i_ch4+1] = np.sqrt((rfghg_sd[i_ch4]+rfch4_sd[i_ch4])**2+ # CH4
                        (rfo3_sd[i_ch4]+rfo3_prime_sd[i_ch4])**2+  # O3
                        0.05**2+                                   # Strat H2O
                        ari_sd[i_ch4]**2+
                        ac_sd[i_ch4]**2)
    error[i_non_ch4+1] = np.sqrt(rfghg_sd[i_non_ch4]**2+
                        rfch4_sd[i_non_ch4]**2+
                        (rfo3_sd[i_non_ch4]+rfo3_prime_sd[i_non_ch4])**2+
                        ari_sd[i_non_ch4]**2+
                        ac_sd[i_non_ch4]**2)
    error[i_aer+1] = np.sqrt(ari_sd[i_aer]**2+
                        ac_sd[i_aer]**2)
    kwargs = {'linewidth':.1,'edgecolor':'k'}
    plt.barh(ybar, pos_co2, width, color=get_chem_col('co2'), label=labels[0],**kwargs)
    plt.barh(ybar, pos_ghg-pos_co2, width, left=pos_co2, color=get_chem_col('WMGHG'), label=labels[1],**kwargs)
    plt.barh(ybar, pos_ch4-pos_ghg, width, left=pos_ghg, color=get_chem_col('ch4'), label=labels[2],**kwargs)
    plt.barh(ybar, pos_o3-pos_ch4, width, left=pos_ch4, color=get_chem_col('o3'), label=labels[3],**kwargs)
    plt.barh(ybar, pos_h2o-pos_o3, width, left=pos_o3, color=get_chem_col('H2O_strat'), label=r'H$_2$O (strat)',**kwargs)
    plt.barh(ybar, pos_aer-pos_h2o, width, left=pos_h2o, color=get_chem_col('ari'), label=labels[4],**kwargs)
    plt.barh(ybar, pos_cloud-pos_aer, width, left=pos_aer, color=get_chem_col('aci'), label=labels[5],**kwargs)
    plt.barh(ybar, neg_ch4, width, color=get_chem_col('ch4'),**kwargs)
    plt.barh(ybar, neg_o3-neg_ch4, width, left=neg_ch4, color=get_chem_col('o3'),**kwargs)
    plt.barh(ybar, neg_aer-neg_o3, width, left=neg_o3, color=get_chem_col('ari'),**kwargs)#'blue,**kwargs')
    plt.barh(ybar, neg_cloud-neg_aer, width, left=neg_aer, color=get_chem_col('aci'),**kwargs)
    plt.errorbar(pos_cloud+neg_cloud,ybar, marker='d', linestyle='None', color='k', label='sum', xerr=error)
    plt.yticks([])#species)

    #plt.yticks(species)

    for i in np.arange(nspec+1):
        #plt.text(-1.55, ybar[i], species[i],  ha='left')#, va='left')
        plt.text(-1.9, ybar[i], species[i],  ha='left')#, va='left')
    plt.title('Components of 1850 to 2014 forcing')
    plt.xlabel(r'W m$^{-2}$')
    plt.xlim(-1.6, 2.0)
    plt.legend(loc='lower right', frameon=False)
    plt.axvline(x=0., color='k', linewidth=0.25)


    sns.despine(fig, left=True)
    plt.show()

    # %%

def get_co2_rf(co2_1850, co2_2014, co2_ra, em_co2, inp_df, n2o_2014, nspec, tot_em_co2):
    # Assume can attributed present day CO2 change by scaling cumulative emissions
    co2 = (em_co2 / tot_em_co2) * (co2_2014 - co2_1850)
    _rfco2 = np.zeros(nspec)
    rfco2 = pd.Series(np.zeros(nspec), index=inp_df.index)
    for ispec in np.arange(nspec):
        _rfco2[ispec] = \
            co2_forcing_AR6(co2_2014, co2_2014 - co2[ispec], n2o_2014) * \
            (1 + co2_ra)
    for spec in np.arange(nspec):
        rfco2[spec] = \
            co2_forcing_AR6(co2_2014, co2_2014 - co2[spec], n2o_2014) * \
            (1 + co2_ra)
    numpy.testing.assert_array_equal(rfco2, _rfco2)
    # co2 contribution from direct co2 emissions
    rfco2_co2 = co2_forcing_AR6(co2_2014, co2_1850, n2o_2014) * (1 + co2_ra) \
                - np.sum(rfco2)  # Subtract off non-co2 carbon contributions
    rfco2_co2
    return co2, rfco2, rfco2_co2


def get_ch4_forcing(ch4, ch4_2014, ch4_ra, ch4_sd, inp_df, inp_sd_df, n2o_2014, nspec):
    _rfch4 = np.zeros(nspec)
    rfch4 = pd.Series(np.zeros(nspec), index=inp_df.index)
    # rfch4 = pd.Series(np.zeros(nspec), index=inp_df.index)
    _rfch4_sd = np.zeros(nspec)
    rfch4_sd = pd.Series(np.zeros(nspec), index=inp_sd_df.index)
    for ispec in np.arange(nspec):
        _rfch4[ispec] = \
            ch4_forcing_AR6(ch4[ispec], ch4_2014, n2o_2014) * \
            (1 + ch4_ra)
        _rfch4_sd[ispec] = \
            ch4_forcing_AR6(ch4[ispec] + ch4_sd[ispec], ch4_2014, n2o_2014) * \
            (1 + ch4_ra) - rfch4[ispec]
    for spec in rfch4.index:  # np.arange(nspec):
        rfch4[spec] = \
            ch4_forcing_AR6(ch4[spec], ch4_2014, n2o_2014) * \
            (1 + ch4_ra)
        rfch4_sd[spec] = \
            ch4_forcing_AR6(ch4[spec] + ch4_sd[spec], ch4_2014, n2o_2014) * \
            (1 + ch4_ra) - rfch4[spec]
    assert_allclose(rfch4, _rfch4)
    # Add in 14% spectral uncertainty
    rfch4_sd=np.sqrt((rfch4*0.14)**2+(rfch4_sd)**2)
    return rfch4, rfch4_sd


def get_ozone_primary_mode(ch4, ch4_1850, ch4_2014, i_ch4, rfo3, rfo3_sd):
    _rfo3perch4 = rfo3[i_ch4] / (ch4_2014 - ch4_1850)  # Use CH4 expt
    rfo3perch4 = rfo3.loc['CH4'] / (ch4_2014 - ch4_1850)  # Use CH4 expt
    assert_allclose(_rfo3perch4, rfo3perch4)
    _rfo3perch4_sd = rfo3_sd[i_ch4] / (ch4_2014 - ch4_1850)  # Use CH4 expt
    rfo3perch4_sd = rfo3_sd.loc['CH4'] / (ch4_2014 - ch4_1850)  # Use CH4 expt
    assert_allclose(_rfo3perch4_sd, rfo3perch4_sd)
    rfo3_prime = rfo3perch4 * (ch4 - ch4_2014)
    rfo3_prime_sd = np.sqrt(
        (rfo3perch4_sd * (ch4 - ch4_2014)) ** 2 +
        # add 15% uncertainty in radiative transfer - from Ragnhild
        (rfo3perch4 * (ch4 - ch4_2014) * 0.15) ** 2)
    return rfo3_prime, rfo3_prime_sd



def getset_ch4(alpha, ch4_2014, i_ch4, i_non_ch4, labs_non_ch4, lifech4, lifech4_sd):
    ch4 = ch4_2014 * (1 + lifech4) ** alpha
    ch4_sd = (ch4 - ch4_2014) * lifech4_sd / lifech4
    _ch4_sd = np.where(lifech4 == 0, 0., ch4_sd)
    ch4_sd[lifech4 == 0] = 0
    numpy.testing.assert_allclose(_ch4_sd, ch4_sd)
    _ch4 = ch4
    # Ch4 due to ch4 is minus sum of non-ch4 terms
    # - ensures total sum of lifetime changes is zero
    _ch4[i_ch4] = \
        ch4_2014 * (1 - np.sum(lifech4[i_non_ch4])) ** alpha
    ch4.loc['CH4'] = \
        ch4_2014 * (1 - np.sum(lifech4[labs_non_ch4])) ** alpha
    assert_allclose(_ch4, ch4)
    ch4_sd.loc['CH4'] = (ch4[i_ch4] - ch4_2014) * \
                        np.sqrt(np.sum(np.square(lifech4_sd[labs_non_ch4]))) / \
                        np.sum(lifech4[i_non_ch4])
    return ch4, ch4_sd



def getset_input_data():
    ncols = 5  # columns in csv file
    nspec = 9  # number of species
    dtype = 'U12' + ', f8' * ncols
    data = np.genfromtxt('attribution_input.csv', delimiter=',', filling_values=0,
                         names=True, dtype=(dtype))
    data_sd = np.genfromtxt('attribution_input_sd.csv', delimiter=',', filling_values=0,
                            names=True, dtype=(dtype))


    _rfo3 = data['o3_rf']
    _rfo3_sd = data_sd['o3_rf_sd']
    _lifech4 = data['lifech4']
    _lifech4_sd = data_sd['lifech4_sd']
    _ari = data['ari']
    _ari_sd = data_sd['ari_sd']
    _ac = data['ac']
    _ac_sd = data_sd['ac_sd']
    _erf = data['erf']
    _erf_sd = data_sd['erf_sd']

    # pandas method:
    fn_input = 'attribution_input.csv'
    fn_input_sd = 'attribution_input_sd.csv'
    inp_df = pd.read_csv(fn_input, index_col=0)
    inp_sd_df = pd.read_csv(fn_input_sd, index_col=0)
    units = inp_df.loc['Units']
    units
    inp_df = inp_df.drop('Units').astype(float)

    inp_sd_df = inp_sd_df.drop('Units').astype(float)


    rfo3 = inp_df['o3_rf']
    rfo3_sd = inp_sd_df['o3_rf_sd']
    lifech4 = inp_df['lifech4']
    lifech4_sd = inp_sd_df['lifech4_sd']
    ari = inp_df['ari']
    ari_sd = inp_sd_df['ari_sd']
    ac = inp_df['ac']
    ac_sd = inp_sd_df['ac_sd']
    erf = inp_df['erf']
    erf_sd = inp_sd_df['erf_sd']
    numpy.testing.assert_allclose(_rfo3[:9], np.array(rfo3))
    numpy.testing.assert_allclose(_lifech4[:9], np.array(lifech4))
    numpy.testing.assert_allclose(_rfo3_sd[:9], np.array(rfo3_sd))


    return inp_df, inp_sd_df, ac, ac_sd, ari, ari_sd, data, erf, erf_sd, lifech4, lifech4_sd, nspec, rfo3, rfo3_sd


# %%
main()