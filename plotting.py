import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]
tau_start = 0.41
tau_stop = 1.41
tau_step = 0.02
tau_start_for_plot = 0.41
# tau_stop_for_plot = 1.41
tau_start_for_comparison = 0.5
# tau_stop_for_comparison = 1.41


def rounddown(x):
    return int(math.floor(x / 100.0)) * 100


def plot_likelihood_heatmap(likelihood_df, tau_start_for_plot, min_log_likelihood):
    """
    Plots the (log) likelihoods of the parameters given the data in a 2D heatmap with speaker_rationality ("SpeakerTau") on the y-axis and listener_rationality ("ListenerTau") on the x-axis.

    :param likelihood_df: Pandas dataframe containing (log) likelihoods. Should contain the following three columns: 'SpeakerTau', 'ListenerTau', and 'LogLikelihood'
    :return: Saves the figure in .pdf format, and shows it
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    #TODO: Should I consider replacing all -inf values in the log likelihood df with NAN? Just to make sure the heatmap is being produced on a fair scale?

    print('')
    print('')
    print("likelihood_df BEFORE SLICING is:")
    print(likelihood_df)

    likelihood_df = likelihood_df[likelihood_df["SpeakerTau"] >= tau_start_for_plot][likelihood_df["ListenerTau"] >= tau_start_for_plot]

    print('')
    print('')
    print("likelihood_df AFTER SLICING is:")
    print(likelihood_df)

    likelihood_df = likelihood_df.pivot(index='SpeakerTau', columns='ListenerTau', values='LogLikelihood')
    ax = sns.heatmap(likelihood_df,  vmin=min_log_likelihood, vmax=-0,
                     xticklabels=likelihood_df.columns.values.round(2),
                     yticklabels=likelihood_df.index.values.round(2))
    plt.title(f"LOG likelihood: {language} + {model}")
    plt.savefig('plots/'+'heatmap_log_likelihood_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()



def plot_bayes_factor_heatmap(bayes_factor_df, tau_start_for_comparison):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(bayes_factor_df,
                     xticklabels=bayes_factor_df.columns.values.round(2),
                     yticklabels=bayes_factor_df.index.values.round(2))
    plt.title(f"Bayes Factors Distance/Person {language}")
    plt.savefig('plots/'+'heatmap_bayes_factors_'+language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()


min_log_value = 0
for language in languages:
    print('')
    print('')
    print(language)
    for model in models:
        print('')
        print('')
        print(model)

        log_likelihood_df = pd.read_pickle(
            'model_fitting_data/' + 'log_likelihood_df_' + language + '_' + model + '_tau_start_' + str(
                tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        print("LOG likelihood_df is:")
        print(log_likelihood_df)

        log_likelihood_array = log_likelihood_df.to_numpy()
        print('')
        print("log_likelihood_array BEFORE NAN_TO_NUM is:")
        print(log_likelihood_array)

        log_likelihood_array = np.nan_to_num(log_likelihood_df, neginf=0)
        print('')
        print("log_likelihood_array AFTER NAN_TO_NUMis:")
        print(log_likelihood_array)

        min = np.min(log_likelihood_array)
        print('')
        print("min is:")
        print(min)

        if min < min_log_value:
            min_log_value = min

        print('')
        print("min_log_value is:")
        print(min_log_value)

        min_log_for_plotting = rounddown(min_log_value)



for language in languages:
    print('')
    print('')
    print(language)
    for model in models:
        print('')
        print('')
        print(model)

        log_likelihood_df = pd.read_pickle(
            'model_fitting_data/' + 'log_likelihood_df_' + language + '_' + model + '_tau_start_' + str(
                tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        print("LOG likelihood_df is:")
        print(log_likelihood_df)

        plot_likelihood_heatmap(log_likelihood_df, tau_start_for_plot, min_log_for_plotting)




for language in languages:
    print('')
    print('')
    print(language)

    bayes_factor_df = pd.read_pickle('model_fitting_data/' + 'bayes_factor_df_' + language + '_' + '_tau_start_' + str(
        tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

    plot_bayes_factor_heatmap(bayes_factor_df, tau_start_for_comparison)