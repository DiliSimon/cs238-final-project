import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    path = 'data'
    directory = Path(path)
    # create subplot to plot mean values
    fig1, axes1 = plt.subplots(5, 5, figsize=(10, 10),constrained_layout=True)
    # flatten indices
    axes1 = axes1.flatten()

    # create subplot to plot standard deviation values
    fig2, axes2 = plt.subplots(5, 5, figsize=(10, 10),constrained_layout=True)
    # flatten indices
    axes2 = axes2.flatten()

    count = 0
    for item in directory.iterdir():
        df = pd.read_csv(item)
        # Plot means of saved data
        axes1[count].plot(df["SARSA_mean"])
        axes1[count].plot(df["Q_mean"])
        axes1[count].legend(['SARSA', "Q-Learning"])
        fig1.suptitle("Running mean of average of scores for 25 separate runs")


        # Plot standard deviation of saved data
        axes2[count].plot(df["SARSA_std"])
        axes2[count].plot(df["Q_std"])
        axes2[count].legend(['SARSA', "Q-Learning"])
        fig2.suptitle("Running mean of standard deviation of scores for 25 separate runs")
        count += 1

    plt.show()  
