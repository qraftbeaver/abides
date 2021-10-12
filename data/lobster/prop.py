import pandas as pd
import numpy as np


with open('AAPL_2012-06-21_34200000_57600000_message_1', 'rb') as f:
    df = pd.read_csv(f)

