import pandas as pd

from ar6_ch6_rcmipfigs.constants import BASE_DIR

# %%
divby = 255
fn = BASE_DIR / 'misc/categorical_colors.csv'
R = 'r'
G = 'g'
B = 'b'


def get_color_df():
    df = pd.read_csv(fn)

    for c in [R, G, B]:
        df[c] = df[c] / divby
    df['color'] = list(zip(df[R], df[G], df[B]))
    df_rein = df.set_index(['type', 'var'])  # columns#set_index(('type','var',))
    # df.close()
    return df_rein


df_rein = get_color_df()

# %%


def get_chem_col(chem):
    """
    :param chem:
    :return:
    """
    df_rein = get_color_df()
    df_chm = df_rein.loc['chem_cat']
    if chem in df_chm.index:
        print(df_chm.loc[chem, 'color'])
        return df_chm.loc[chem, 'color']
    else:
        return None


def get_scn_col(scn):
    """

    :param scn:
    :return:
    """
    df_rein = get_color_df()
    df_scn = df_rein.loc['ssp_cat_2']
    if scn in df_scn.index:
        return df_scn.loc[scn, 'color']
    else:
        return None
