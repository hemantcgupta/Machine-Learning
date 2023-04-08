# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:13:14 2023

@author: Hemant
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Data Collection, EDA and Dat Processing
# =============================================================================
df = pd.read_csv(r'Car Price Data.csv')

df.shape
df.info()





















