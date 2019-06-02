import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(font_scale=1.3, style='white')


# if __name__ == "__main__":
#     fontsize=26
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     df = pd.read_json("hyperparameter_search_results/vampire_npmi_classifier_search.jsonl", lines=True)
#     sns.regplot(df['best_validation_npmi'], df['best_validation_accuracy'])
#     ax.set_xlabel("Best validation NPMI", )
#     ax.set_ylabel("Best validation accuracy", )
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize) 
#     for label in ax.xaxis.get_ticklabels()[::2]:
#         label.set_visible(False)

#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize)
#     ax.set_ylim([0.60, 0.85])
#     plt.savefig("results/regplot_figure.pdf", dpi=300)


# if __name__ == "__main__":
#     fontsize=26
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     df = pd.read_json("hyperparameter_search_results/vampire_npmi_classifier_search.jsonl", lines=True)
#     sns.regplot(df['best_validation_npmi'], df['best_validation_nll'])
#     ax.set_xlabel("Best validation NPMI", )
#     ax.set_ylabel("Best validation NLL", )
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize) 
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize)
#     for label in ax.xaxis.get_ticklabels()[::2]:
#         label.set_visible(False)
#     # ax.set_ylim([0.60, 0.85])
#     plt.savefig("results/regplot_figure_1.pdf", dpi=300)


# if __name__ == '__main__':
#     fontsize=23
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     df = pd.read_json("hyperparameter_search_results/clf_search.jsonl", lines=True)
#     sns.boxplot(df['model.encoder.architecture.type'],df['best_validation_accuracy'])  
#     ax.set_xticklabels(["CNN", "LSTM", "Averaging"], fontsize=26)                                                                                                                                                                                          
#     ax.set_xlabel('Classifier Encoder', fontsize=26) 
#     ax.set_ylabel("Validation Accuracy", fontsize=26)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize) 
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(fontsize)
#     plt.savefig("results/clf_accuracy_figure.pdf", dpi=300)

if __name__ == '__main__':
    import matplotlib.gridspec as gridspec
    fig, ax = plt.subplots(2, 2)
    
    ax1 = ax[0,0]
    ax2 = ax[0,1]
    ax3 = ax[1,0]
    ax4 = ax[1,1]

    df = pd.read_json("hyperparameter_search_results/vampire_npmi_classifier_search.jsonl", lines=True)
    sns.regplot(df['best_validation_npmi'], df['best_validation_nll'], ax=ax1, color='black')
    ax1.set_xlabel("NPMI")
    ax1.set_ylabel("NLL")
    # for tick in ax3.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize) 
    # for tick in ax3.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize)
    ax1.xaxis.set_ticks([0.06, 0.14])
    ax1.set_ylim([820, 900])
    ax1.yaxis.set_ticks([840, 860, 880])


    ax1.text(-0.1, 1.15, "A", transform=ax1.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

    df = pd.read_json("hyperparameter_search_results/vampire_npmi_classifier_search.jsonl", lines=True)
    sns.regplot(df['best_validation_npmi'], df['best_validation_accuracy'], ax=ax2)
    ax2.set_xlabel("NPMI")
    ax2.set_ylabel("Accuracy")
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize)
    ax2.set_ylim([0.60, 0.85])
    ax2.xaxis.set_ticks([0.06, 0.14])
    ax2.yaxis.set_ticks([0.7, 0.8])
    
    # for tick in ax2.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize) 
    # for label in ax2.xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)
    
    ax2.text(-0.1, 1.15, "B", transform=ax2.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')


    df = pd.read_json("hyperparameter_search_results/vampire_nll_classifier_search.jsonl", lines=True)

    sns.regplot(df['best_validation_nll'], df['best_validation_accuracy'], ax=ax3)
    ax3.set_xlabel("NLL")
    ax3.set_ylabel("Accuracy")
    # for tick in ax4.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize) 
    # for tick in ax4.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize) 
    # for label in ax4.xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)
    ax3.text(-0.1, 1.15, "C", transform=ax3.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

    df = pd.read_json("hyperparameter_search_results/vampire_nll_classifier_search.jsonl", lines=True)
    df1 = pd.read_json("hyperparameter_search_results/vampire_npmi_classifier_search.jsonl", lines=True)
    master = pd.concat([df, df1], 0)
    master['trainer.validation_metric_y'] = master['trainer.validation_metric_y'].fillna('-nll')
    sns.boxplot(master['trainer.validation_metric_y'], master['best_validation_accuracy'], ax=ax4, order = ["+npmi", "-nll"])  
    ax4.set_xticklabels(["NPMI", "NLL"])                                                                                                                                                                                          
    ax4.set_xlabel('Criterion') 
    ax4.set_ylabel("Accuracy")
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize) 
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(fontsize)
    ax4.set_ylim([0.4, 0.9])
    ax4.yaxis.set_ticks([0.6, 0.8])
    ax4.text(-0.1, 1.15, "D", transform=ax4.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

    plt.tight_layout()


    

    
    


    
    plt.savefig("figure_4.pdf", dpi=300)

