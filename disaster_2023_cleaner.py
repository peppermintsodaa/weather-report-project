import pandas as pd
import re
import difflib
import matplotlib.pyplot as plt
import numpy as np

# read excel file
disaster_2023 = pd.read_excel('datasets/disaster-mapper-data-21-03-2023.xlsx')

# eliminate all header rows
disaster_2023.dropna(subset=['Zone'], inplace = True)

# strip whitespace off all columns
disaster_2023.rename({'Event ':'Event'}, axis = 1, inplace = True)

# drop all unnecessary columns
disaster_2023.drop(columns = ['Fatalities', 'Injured', 'URL', 'Description', 'Source(s)'], inplace = True)