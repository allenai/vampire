import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(font_scale=1.3, style='white')


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(6,3))
    df = pd.read_csv("~/Downloads/run-vampire_AG_log_validation-tag-npmi.csv") 
    df1 = pd.read_csv("~/Downloads/run-vampire_AG_log_validation-tag-nll.csv")
    sns.lineplot(df1['Step'], df1['Value'], ax=ax[0])
    sns.lineplot(df['Step'], df['Value'], ax=ax[1])
    ax[1].set_xlabel("Epoch")
    ax[0].set_ylabel("NLL")
    ax[1].set_ylabel("NPMI")
    plt.tight_layout()
    plt.savefig("curves.pdf")