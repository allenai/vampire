import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

class PlotUtils(object):
    @staticmethod
    def initialize_figure(subplotsize, figsize):
        fig, axes = plt.subplots(subplotsize[0],
                                    subplotsize[1],
                                    figsize=figsize,
                                    sharex=True,
                                    sharey=True)
        return fig, axes

    @staticmethod
    def save_figure(fig, output_path: str):
        fig.savefig(output_path, dpi=300)


class Dataset(object):
    def __init__(self, data_path: str, dataset_name: str, ignore_experiment_name=False, dropna=True):
        self._data_path = data_path
        self.dataset_name = dataset_name
        self.df = pd.read_csv(self._data_path)
        self.throttles = [200, 500, 2500, 5000, 10000]
        self.num_seeds = len(self.df.env_SEED.unique())
        if ignore_experiment_name:
            self.df = self.df.drop(['Experiment_Name'], 1)
        if dropna:
            self.df = self.df.dropna()

    @classmethod
    def from_file(cls, file_path: str, dataset_name: str, ignore_experiment_name: bool, dropna: bool):
        return cls(file_path, dataset_name, ignore_experiment_name, dropna)


class GenerateBaselinePlots(PlotUtils):

    def __init__(self, data_path: str, dataset_name: str, ignore_experiment_name=False, dropna=True):
        self.data = Dataset.from_file(data_path, dataset_name, ignore_experiment_name, dropna)

    def boxplot_accuracy_over_throttles(self, by_clf=False, save_path=None, show=True):
        if by_clf:
            classifiers = self.data.df.env_CLASSIFIER.unique()
            fig, axes = self.initialize_figure((len(classifiers), 1), (7, 15))
            for classifier, axis in zip(classifiers, axes):
                sub_df = self.data.df.loc[self.data.df.env_CLASSIFIER == classifier]
                sns.boxplot(sub_df['env_THROTTLE'],
                            sub_df.metric_best_validation_accuracy,
                            ax=axis)
                axis.set(ylabel='validation accuracy (%)')
                axis.set_title(classifier)
                axis.set_xlabel('')
            axes[-1].set(xlabel='throttle')
        else:
            fig, axis = self.initialize_figure((1, 1), (8, 6))
            sns.boxplot(self.data.df['env_THROTTLE'],
                        self.data.df['metric_best_validation_accuracy'])
            axis.set(xlabel='throttle', ylabel='validation accuracy')
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

    def pdf_accuracy_over_throttle(self, throttle=200, save_path=None, show=True):
        if throttle not in self.throttles:
            raise Exception(f"throttle {throttle} is not available")
        fig, axis = self.initialize_figure((1, 1), (8, 6))
        classifiers = self.data.df.env_CLASSIFIER.unique()
        for classifier in classifiers:
            sub_df = self.data.df.loc[(self.data.df.env_CLASSIFIER == classifier) &
                                        (self.data.df.env_THROTTLE == throttle)]
            sns.distplot(sub_df.metric_best_validation_accuracy,
                         hist=False,
                         norm_hist=True,
                         label=classifier)
        axis.set_title("PDF of {} classification accuracy: {} training samples ({} trials across {} seeds)".format(self.data.dataset_name,
                                                                                                                   throttle,
                                                                                                                   sub_df.shape[0],
                                                                                                                   self.data.num_seeds))
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)


class GenerateVAEPlots(PlotUtils):
    def __init__(self, data_path: str, dataset_name: str, ignore_experiment_name=False, dropna=True):
        self.data = Dataset.from_file(data_path, dataset_name, ignore_experiment_name, dropna)

    def nll_npmi_scatter(self, show=True, save_path=None):
        fig, axis = self.initialize_figure((1, 1), (8, 6))
        plt.scatter(self.data.df.metric_best_validation_npmi, self.data.df.metric_best_validation_nll)
        axis.set_xlabel("npmi")
        axis.set_ylabel("nll")
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

    def nll_npmi_convergence_time(self, show=True, save_path=None):
        fig, axis = self.initialize_figure((1, 1), (8, 6))
        sns.boxplot(self.data.df.env_VALIDATION_METRIC, self.data.df.metric_best_epoch, order=['-nll', '+npmi'])
        axis.set_xlabel("validation metric")
        axis.set_ylabel("best epoch")
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

    def global_npmi_density_plot(self, show=True, save_path=None):
        fig, axis = self.initialize_figure((1, 1), (8, 6))
        sns.distplot(self.data.df.loc[self.data.df['env_VALIDATION_METRIC'] == '+npmi']['metric_best_validation_nll'],
                     hist=False,
                     norm_hist=True)
        axis.set_ylabel('density')
        axis.set_xlabel("nll")
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

    def global_nll_density_plot(self, show=True, save_path=None):
        fig, axis = self.initialize_figure((1, 1), (8, 6))
        sns.distplot(self.data.df.loc[self.data.df['env_VALIDATION_METRIC'] == '+npmi']['metric_best_validation_nll'],
                     hist=False,
                     norm_hist=True)
        axis.set_ylabel('density')
        axis.set_xlabel("nll")
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

    def boxplot_effect_of_architecture_on_npmi(self, architecture_field_name: str, show: bool=True, save_path: str=None):
        fig, _ = self.initialize_figure((1, 1), (8, 6))
        sns.boxplot(self.data.df.loc[self.data.df.env_VALIDATION_METRIC == "+npmi"][architecture_field_name],
                    self.data.df.loc[self.data.df.env_VALIDATION_METRIC == "+npmi"].metric_best_validation_npmi)
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)


class GeneratePretrainPlots(PlotUtils):

    def __init__(self, **kwargs):
        self.datasets = []
        self.dataset_name = kwargs.pop('dataset_name')
        for condition, file in kwargs.items():
            data = Dataset.from_file(file, self.dataset_name, ignore_experiment_name=True, dropna=False)
            data.df['condition'] = condition
            self.datasets.append(data.df)
        if len(self.datasets) > 1:
            self.master = pd.concat(self.datasets, 0)
        else:
            self.master = self.datasets[0]

    def boxplot_global_compare_at_throttle(self, throttle=200, by_clf=False, show=True, save_path=None):
        if by_clf:
            classifiers = self.master.env_CLASSIFIER.unique()
            fig, axes = self.initialize_figure((len(classifiers), 1), (7, 15))
            for classifier, axis in zip(classifiers, axes):
                sub_df = self.master.loc[(self.master.env_CLASSIFIER == classifier) & (self.master.env_THROTTLE == throttle)]
                sns.boxplot(sub_df.condition, sub_df.metric_best_validation_accuracy, ax=axis)
                axis.set(ylabel='validation accuracy (%)')
                axis.set_title("{}, {} docs".format(classifier, throttle))
                axis.set_xlabel('')
            axes[-1].set(xlabel='condition')
        else:
            fig, axis = self.initialize_figure((1, 1), (8, 6))
            sub_df = self.master.loc[self.master.env_THROTTLE == throttle]
            sns.boxplot(sub_df.condition, sub_df.metric_best_validation_accuracy)
            axis.set(xlabel='condition', ylabel='validation accuracy')
            axis.set_title("Downstream Classification with Pre-training, {} {} documents".format(self.dataset_name, throttle))
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

    def pdf_accuracy_over_throttle_and_conditions(self, throttle=200, save_path=None, show=True):
        classifiers = self.master.env_CLASSIFIER.unique()
        conditions = self.master.condition.unique()
        fig, axes = self.initialize_figure((len(classifiers), 1), (8, 6))
        for axis, classifier in zip(axes, classifiers):
            total_size = 0
            for condition in conditions:
                sub_df = self.master.loc[(self.master.env_CLASSIFIER == classifier) &
                                        (self.master.env_THROTTLE == throttle) &
                                        (self.master.condition == condition)]
                total_size += sub_df.shape[0]
                sns.distplot(sub_df.metric_best_validation_accuracy.dropna(),
                                hist=False,
                                norm_hist=True,
                                label="{}_{}".format(classifier, condition),
                                ax=axis)
            
            axis.set_title("{} training samples ({} trials)".format(throttle,
                                                                    total_size
                                                                    ))
            axis.set_xlabel('')
        axes[-1].set(xlabel='validation accuracy (%)')
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

    def boxplot_effect_of_architecture_on_accuracy(self, architecture_field_name: str, throttle: int=200, by_clf: bool = False, show: bool=True, save_path: str=None):
        if by_clf:
            classifiers = self.master.env_CLASSIFIER.unique()
            fig, axes = self.initialize_figure((len(classifiers), 1), (7, 15))
            for classifier, axis in zip(classifiers, axes):
                sub_df = self.master.loc[(self.master.env_CLASSIFIER == classifier) &
                                         (self.master.env_THROTTLE == throttle)]
                sns.boxplot(sub_df[architecture_field_name],
                            sub_df.metric_best_validation_accuracy, ax=axis)
                axis.set(ylabel='validation accuracy (%)')
                axis.set_title("{}, {} docs".format(classifier, throttle))
                axis.set_xlabel('')
            axes[-1].set(xlabel=architecture_field_name)
        else:
            fig, _ = self.initialize_figure((1, 1), (8, 6))
            sns.boxplot(self.master.loc[self.master.env_THROTTLE == throttle][architecture_field_name],
                        self.master.loc[self.master.env_THROTTLE == throttle].metric_best_validation_accuracy)
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)
    
    def heatmap_on_accuracy(self, architecture_field_name_1: str, architecture_field_name_2: str, throttle: int=200, by_clf: bool = False, show: bool=True, save_path: str=None):
        import ipdb; ipdb.set_trace()
        master = self.master[[architecture_field_name_1, architecture_field_name_2, 'metric_best_validation_accuracy']].pivot(architecture_field_name_1, architecture_field_name_2, "metric_best_validation_accuracy")
        if by_clf:
            classifiers = master.env_CLASSIFIER.unique()
            fig, axes = self.initialize_figure((len(classifiers), 1), (7, 15))
            for classifier, axis in zip(classifiers, axes):
                sub_df = master.loc[(master.env_CLASSIFIER == classifier) &
                                    (master.env_THROTTLE == throttle)]
                sns.heatmap(sub_df, annotate=True, ax=axis)
                axis.set_title("{}, {} docs".format(classifier, throttle))
        else:
            fig, _ = self.initialize_figure((1, 1), (8, 6))
            master = master.loc[(master.env_CLASSIFIER == classifier) &
                                    (master.env_THROTTLE == throttle)]
            sns.heatmap(master, annotate=True, ax=axis)
            axis.set_title("{} docs".format(throttle))
        if show:
            plt.show()
        if save_path:
            self.save_figure(fig, save_path)

if __name__ == '__main__':
    DATA_DIR = "/Users/suching/Github/vae/results/csv/vae"
    
    # GBP = GenerateBaselinePlots(DATA_DIR,
    #                             "AGNEWS",
    #                             sep=',',
    #                             ignore_experiment_name=True,
    #                             dropna=True)
    # GBP.boxplot_accuracy_over_throttles(by_clf=True)
    PRETRAINED_DIR = "/Users/suching/Github/vae/results/csv/pretrained"
    BASELINE_DIR = "/Users/suching/Github/vae/results/csv/baseline"
    VAE_DIR = "/Users/suching/Github/vae/results/csv/vae"
    PRETRAINED_DIR = "/Users/suching/Github/vae/results/csv/pretrained"

    DATASETS = ["IMDB", "AGNEWS"]
    GPP = GeneratePretrainPlots(baseline=os.path.join(BASELINE_DIR, "IMDB", "master.csv"),
                                baseline_with_npmi_vae=os.path.join(PRETRAINED_DIR, "+npmi_vae", "master.csv"),
                                dataset_name="IMDB")
    GPP.pdf_accuracy_over_throttle_and_conditions(throttle=10000)
    # GPP.heatmap_on_accuracy("env_DROPOUT", "env_VAE_DROPOUT", throttle=200, by_clf=False)                      
    # for dataset in ["IMDB", "AGNEWS"]:
    #     GVP = GenerateVAEPlots(os.path.join(DATA_DIR, dataset + ".csv"),
    #                            dataset,
    #                            ignore_experiment_name=True,
    #                            dropna=True)
    #     GVP.boxplot_effect_of_architecture_on_npmi("env_VAE_HIDDEN_DIM")

