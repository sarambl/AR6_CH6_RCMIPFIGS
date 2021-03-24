import os
import ar6_ch6_rcmipfigs
from pathlib import Path

# %%
BASE_DIR = Path(os.path.dirname(ar6_ch6_rcmipfigs.__file__))
print(BASE_DIR)
# %%
INPUT_DATA_DIR = BASE_DIR/ 'data_in'
OUTPUT_DATA_DIR = BASE_DIR / 'data_out'
print(INPUT_DATA_DIR)
RESULTS_DIR = BASE_DIR/'results'
# %%