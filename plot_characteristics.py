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

import pandas as pd
from matplotlib import pyplot as plt


if __name__ == '__main__':
    df = pd.concat([pd.read_csv('motor_characteristics_withbrake_2450kv_.csv')])
    df.sort_values(by='power', inplace=True)
    plt.plot(df['power'], -df['thrust'])
    plt.xlabel('power')
    plt.ylabel('thrust (in grams)')
    plt.title('BLDC thrust characteristics plot')
    plt.savefig('motor_characteristics_plot_2450_.png', dpi=300)

