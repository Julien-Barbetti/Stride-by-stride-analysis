# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:59:32 2025

@author: Utilisateur
"""

import pandas as pd
import numpy as np
from typing import Literal
from src.core.find_peaks import trouve_pics

def detect_top_crank_position(crank_signal : np.array, fs : float, foot: Literal["left", "right"]) -> pd.DataFrame:


    pic_R = trouve_pics(-1*crank_signal, tolerance = 5)

    top_index = sorted(pic_R["index"].tolist())



    return pd.DataFrame({
        "cycle": np.arange(len(top_index) - 1),
        "foot": [foot] * (len(top_index) - 1),
        "time": np.array(top_index[:-1]) / fs,
        "indice": top_index[:-1]
    })




