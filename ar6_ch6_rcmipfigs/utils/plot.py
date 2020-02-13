import numpy as np
import pyam
import seaborn as sns
from matplotlib import pyplot as plt

from ar6_ch6_rcmipfigs.utils.misc_func import climatemodel


def plot_available_out(db, variables, scenarios, figsize=[30, 30]):
    """
    Plots specified variables and scenarios to get overview over available data
    :param db: scmdata.dataframe input data
    :param variables: variable list to be plotted
    :param scenarios: scenario list to be plotted
    :param figsize: figure size
    :return:
    """
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(len(variables), len(scenarios), figsize=figsize, sharex=True)
    j = -1
    for var in variables:
        j += 1
        i = -1
        for scn in scenarios:
            i += 1
            ax = axs[j, i]
            _db = db.filter(variable=var, scenario=scenarios)
            _df = _db.filter(variable=var, scenario=scn)  # 'ssp119')#.timeseries()#[variables_erf[0]]
            __df = _df.timeseries().transpose()  # .plot(ax=ax)
            for model in __df.transpose().reset_index()[climatemodel]:
                # if model in __df:
                __df.xs(model, level=climatemodel, axis=1).plot(ax=ax)  # , label='sdf')
            ax.set_title(scn)
            ax.set_ylabel('W/m2, %s' % var.split('|')[-1])
            ax.set_xlim(['1850', '2100'])
            ax.legend(__df.transpose().reset_index()[climatemodel], frameon=False)
        # ax.set_ylim([-4,4])


def get_cmap_dic(keys, palette='colorblind'):
    cols = sns.color_palette(palette, n_colors=len(keys))
    colordic = {}  # 'NorESM1-LM':col}
    for model, col in zip(keys, cols):
        colordic[model] = col
    return colordic


def get_ls_dic(keys):
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']#(0, (1, 1))]
    odic = {}
    if len(keys) > len(linestyles):
        print('Warning: too many keys')
    for key, ls in zip(keys, linestyles):
        odic[key] = ls
    return odic


def trans_scen2plotlabel(label):
    if label == ssp370low_nn:
        return ssp370low_on
    else:
        return label


color_map_scenarios_base = {
    "ssp119": "AR6-SSP1-1.9",
    "ssp126": "AR6-SSP1-2.6",
    "ssp245": "AR6-SSP2-4.5",
    "ssp370": "AR6-SSP3-7.0",
    "ssp370-lowNTCF": "AR6-SSP3-LowNTCF",
    "ssp370-lowNTCF-aerchemmip": "AR6-SSP3-LowNTCF",
    "ssp434": "AR6-SSP4-3.4",
    "ssp460": "AR6-SSP4-6.0",
    "ssp585": "AR6-SSP5-8.5",
    "ssp534-over": "AR6-SSP5-3.4-OS",
    # "historical": "black",
    "rcp26": "AR5-RCP-2.6",
    "rcp45": "AR5-RCP-4.5",
    "rcp60": "AR5-RCP-6.0",
    "rcp85": "AR5-RCP-8.5",
    # "historical-cmip5": "tab:gray",
}

# %%
from ar6_ch6_rcmipfigs.constants import BASE_DIR


# %%
def get_scenario_c_dic(new=True):
    if new:
        path_cf = BASE_DIR + '/misc/ssp_cat_2.txt'
        rgb_data_in_the_txt_file = np.loadtxt(path_cf)
        colormap_dic = {}
        for scn, col in zip(scenario_list, rgb_data_in_the_txt_file):
            colormap_dic[scn] = tuple([a / 255. for a in col])
        colormap_dic[ssp370low_nn] = colormap_dic[ssp370low_on]
        colormap_dic['historical'] = 'black'
        return colormap_dic

    colormap_dic = {}
    ipccdic = pyam.plotting.PYAM_COLORS
    # print(ipccdic)
    for each in color_map_scenarios_base:
        # print(key)
        colormap_dic[each] = ipccdic[color_map_scenarios_base[each]]
    colormap_dic['historical'] = 'black'
    # print(each)
    return colormap_dic


def get_scenario_ls_dic():
    c_dic = get_scenario_c_dic()
    ls_dic = {}
    for key in c_dic:
        ls_dic[key] = 'solid'
    ls_dic[ssp370low_nn] = 'dashed'
    return ls_dic


scenario_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp370-LowNTCF',
                 'ssp435', 'ssp460', 'ssp534os', 'ssp585']
ssp370low_nn = "ssp370-lowNTCF-aerchemmip"
ssp370low_on = 'ssp370-LowNTCF'
