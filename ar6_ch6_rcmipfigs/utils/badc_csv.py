import sys

import numpy as np
import pandas as pd
from ar6_ch6_rcmipfigs.constants import BASE_DIR

from pathlib import Path

path_FaIR_header_general_info = Path(BASE_DIR) / 'misc/badc_header_FaIR_model.csv'
path_FaIR_warming_header_general_info = Path(BASE_DIR) / 'misc/badc_header_FaIR_model_warming.csv'
path_FaIR_hist_header_general_info = Path(BASE_DIR) / 'misc/badc_header_FaIR_model_hist.csv'

# %%
fp_example = 'ar6_ch6_rcmipfigs/data_in/SSPs_badc-csv/ERF_ssp119_1750-2500.csv'
fp_example_test = 'ar6_ch6_rcmipfigs/data_in/SSPs_badc-csv/ERF_ssp119_1750-2500_test.csv'

fp_orig_example = 'ar6_ch6_rcmipfigs/data_in/SSPs/ERF_ssp119_1750-2500.csv'


# %%
def get_header_length(fp):
    """
    Finds the header length in a BADC csv file
    :param fp: file path
    :return:
    """
    cnt_data = 0
    with open(fp) as f:
        line = f.readline()
        cnt = 1
        while line:
            l_sp = line.split(',')
            if l_sp[0].strip() == 'data':
                cnt_data = cnt
                break
            line = f.readline()
            cnt += 1

    return cnt_data


# %%
def read_csv_badc(fp, **kwargs):
    # %%
    if kwargs is None:
        kwargs = {'index_col': 0}
    length_header = get_header_length(fp)
    if 'header' in kwargs.keys():
        hd = kwargs['header']
        if type(hd) is list:
            hd: list
            length_header = list(np.array(hd) + length_header - hd[0])
        del kwargs['header']
    df = pd.read_csv(fp, skipfooter=1, header=length_header, **kwargs, engine='python')
    if df.index[-1] == 'end_data':
        df = df.drop('end_data', axis=0)

    # %%
    return df


# %%
def get_global_FaIR(fp=path_FaIR_header_general_info):
    df = pd.read_csv(fp, header=None)
    df_glob = df[df.iloc[:, 1] == 'G']
    return df_glob
    # %%


def get_variable_FaIR(fp=path_FaIR_header_general_info):
    df = pd.read_csv(fp, header=None)
    df_vars = df[df.iloc[:, 1] != 'G']
    return df_vars


# %%
def write_badc_header(
        fp_orig,
        fp_out,
        add_global_info,
        # variable_dic,
        # read_csv_kwargs=None,
        fp_global_default=path_FaIR_header_general_info,
        fp_var_default=path_FaIR_header_general_info,
        default_unit='W/m2'
):
    # fp_global_default = path_FaIR_header_general_info
    # add_global_info = [['comments','G','Scenario: SSP1-1.9'],]
    # fp_var_default = path_FaIR_header_general_info
    # read_csv_kwargs=None
    # fp_orig = fp_orig_example

    df_glob = get_global_FaIR(fp=fp_global_default)
    df_var = get_variable_FaIR(fp=fp_var_default)
    df_glob.head()
    # if global_info_dic is not None:
    #    df_glob

    df_extra_glob = pd.DataFrame(add_global_info)
    df_glob = df_glob.append(df_extra_glob)

    df_orig = pd.read_csv(fp_orig, index_col=0, header=None)
    var_labs = [df_orig.index[0]] + list(df_orig.iloc[0, :])

    # var_labs

    _df = pd.read_csv(fp_orig, index_col=None, header=None)
    _df
    df_header = df_glob
    for var in var_labs:
        lines = df_var.iloc[:, 1] == var
        df_header = df_header.append(df_var[lines])

        if len(df_var[lines]) == 0:
            # no set info:
            fix = pd.DataFrame([['metdb_short_name', var, var, default_unit],
                                ['long_name', var, var, default_unit],
                                ['type', var, 'float', ], ], )
            df_header = df_header.append(fix)

    df_header = df_header.append(pd.DataFrame(['data']))

    df_out = df_header.append(_df)
    df_out = df_out.append(pd.DataFrame(['end_data']))
    df_out.to_csv(fp_out, header=False, index=False)

    return df_out


# %%


def redo_SSPs_to_badc_csv(path_orig, path_out, ):
    """

    :param path_orig:
    :param path_out:
    :return:
    """
    # path_orig = Path(BASE_DIR) / 'data_in/SSPs/'
    # path_out = Path(BASE_DIR) / 'data_in/SSPs_badc-csv/'
    path_out.mkdir(exist_ok=True, parents=True)
    for f in list(path_orig.glob('*.csv')):
        filename = f.name

        f_out = path_out / filename
        # print(f_out)
        fn_parts = filename.split('_')
        scenario_name = ' '.join(fn_parts[1:-1])
        add_global_info = [['comments', 'G', f'Scenario: {scenario_name}'], ]
        write_badc_header(
            f,
            f_out,
            add_global_info,
            # variable_dic,
            # read_csv_kwargs=None,
            # fp_global_default = path_FaIR_header_general_info,
            # fp_var_default = path_FaIR_header_general_info
        )
    return


def make_badc_csvs_for_slcf_warming_ranges(
        fn_orig=(BASE_DIR / 'data_in/chris_slcf_warming_ranges.csv'),
        fn_base='slcf_warming_ranges',
        path_out=(BASE_DIR / 'data_in_badc_csv' / 'slcf_warming_ranges'),
):
    path_out.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_csv(fn_orig, index_col=0)
    scenario_ls = df_orig.loc[:, 'scenario'].unique()
    dic_scn_data = dict()
    if len(df_orig['base_period'].unique()) > 1:
        print('ERROR: Different base_periods found! ')
        sys.exit()
    df_orig = df_orig.drop(['base_period'], axis=1)
    pcts = ['p05', 'p16', 'p50', 'p84', 'p95']

    for scn in scenario_ls:
        dic_scn_data[scn] = {}
        df_sub_scn = df_orig[df_orig['scenario'] == scn]
        df_sub_scn = df_sub_scn.drop('scenario', axis=1)
        for p in pcts:
            df_sub_scn_p = df_sub_scn[['year', 'forcing', p]]  # .set_index('year')
            df_n = df_sub_scn_p.set_index(['forcing', 'year']).unstack()[p].T

            fn_sub = fn_base + f'_{scn}_{p}.csv'
            fp = path_out / fn_sub
            df_n.to_csv(fp)
            add_global_info = [['comments', 'G', f'Scenario: {scn}, percentile: {p}'], ]
            write_badc_header(
                fp,
                fp,
                add_global_info,
                # variable_dic,
                # read_csv_kwargs=None,
                fp_global_default=path_FaIR_warming_header_general_info,
                # fp_var_default = path_FaIR_header_general_info
            )

        #    df_sub_scn_p = df_sub_scn
    return


def get_add_global_from_dic(dic_head, add_global_comments=None):
    add_global = [[key, 'G', dic_head[key]] for key in dic_head.keys()]
    if add_global_comments is not None:
        add_global = add_global + add_global_comments
    return add_global

# %%


# %%
