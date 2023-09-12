"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

from glob import glob


from matplotlib import pyplot as plt
import pandas as pd


if __name__ == '__main__':
    fpath = glob('a22*')[0]
    df = pd.read_csv(fpath)

    x = df['Speed %']
    power_cols = [c for c in df.columns if c in ['Theoretical Power (W)', 'Actual Power (W)', 'Power Loss (W)']]

    ax, fig = plt.subplots()




    k = 0






