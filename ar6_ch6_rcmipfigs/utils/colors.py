import pandas as pd
from ar6_ch6_rcmipfigs.constants import BASE_DIR

# %%
divby = 255
fn = BASE_DIR / 'misc/categorical_colors.csv'

df = pd.read_csv(fn)
R ='r'
G='g'
B='b'
for c in [R,G,B]:
    df[c]=df[c]/divby
df['color']=list(zip(df[R],df[G],df[B]))
df_rein = df.set_index(['type','var'])#columns#set_index(('type','var',))

# %%
import matplotlib.pyplot as plt
def get_chem_col(chem):
    """
    :param chem:
    :return:
    """
    if chem in df_rein.loc['chem_cat'].index:
     return df_rein.loc[('chem_cat',chem),'color'].values[0]
    else:
        return None



