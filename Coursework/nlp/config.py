from typing import Union, Any

import pandas as pd
import numpy as np


DATA_FOLDER = "data"
DATA_PCL_NAME = "dontpatronizeme_pcl.csv"
DATA_CATEGORIES_NAME = "dontpatronizeme_categories.csv"
TRAIN_ID = "train_semeval_parids-labels.csv"
DEV_ID = "dev_semeval_parids-labels.csv"
BASELINE_DF_NAME = "df_baseline.csv"
MODEL_FOLDER = "models"
MODEL_NAME = "nlp_model.pt"
BASELINE_PATH = "outputs/"
PLOT_FOLDER = "plots"

# Type
Array_like = Union[list, np.ndarray, pd.Series, pd.DataFrame, Any]
