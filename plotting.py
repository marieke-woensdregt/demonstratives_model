import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]
tau_start = 0.5
tau_stop = 10.
tau_step = 0.25

tau_start_for_plot = 0.5
# tau_stop_for_plot = 1.41
tau_start_for_comparison = 0.5
# tau_stop_for_comparison = 1.41


def rounddown(x):
    return int(math.floor(x / 100.0)) * 100


def calc_min_log_likelihood(words, object_positions, listener_positions):
    max_prob = 1.0
    max_log_prob = np.log(max_prob)
    likelihood_product = 1.0
    max_log_likelihood_sum = np.log(1.0) # The first probability should be multiplied with 1.0, which is equivalent to 0.0 in log-space
    for object_pos in object_positions:
        for listener_pos in listener_positions:
            likelihood_product *= max_prob
            max_log_likelihood_sum += max_log_prob  # multiplication in probability space is equivalent to addition in log-space
    return likelihood_product, max_log_likelihood_sum




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



def plot_which_model_wins(distance_wins_df, tau_start_for_comparison):
    fig, ax = plt.subplots(figsize=(11, 9))

    vmap = {0:'person', 1:'distance'}
    n = len(vmap)

    cmap = sns.color_palette("colorblind", n)
    ax = sns.heatmap(distance_wins_df,
                     xticklabels=distance_wins_df.columns.values.round(2),
                     yticklabels=distance_wins_df.index.values.round(2),
                     cmap=cmap)

    # Get the colorbar object from the Seaborn heatmap
    colorbar = ax.collections[0].colorbar
    # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(n)])
    colorbar.set_ticklabels(list(vmap.values()))

    plt.title(f"Which model 'wins': {language}")
    plt.savefig('plots/'+'heatmap_which_model_wins_'+language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()


min_log_value_two_system = 0
for language in ["English", "Italian"]:
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

        if min < min_log_value_two_system:
            min_log_value_two_system = min

        print('')
        print("min_log_value_two_system is:")
        print(min_log_value_two_system)

        min_log_value_two_system = rounddown(min_log_value_two_system)




min_log_value_three_system = 0
for language in ["Portuguese", "Spanish"]:
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

        if min < min_log_value_three_system:
            min_log_value_three_system = min

        print('')
        print("min_log_value_three_system is:")
        print(min_log_value_three_system)

        min_log_value_three_system = rounddown(min_log_value_three_system)



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

        if language == "English" or language == "Italian":
            plot_likelihood_heatmap(log_likelihood_df, tau_start_for_plot, min_log_value_two_system)
        elif language == "Portuguese" or language == "Spanish":
            plot_likelihood_heatmap(log_likelihood_df, tau_start_for_plot, min_log_value_three_system)




for language in languages:
    print('')
    print('')
    print(language)

    bayes_factor_df = pd.read_pickle('model_fitting_data/' + 'bayes_factor_df_' + language + '_' + '_tau_start_' + str(
        tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

    plot_bayes_factor_heatmap(bayes_factor_df, tau_start_for_comparison)

    distance_wins_df = pd.read_pickle('model_fitting_data/' + 'distance_wins_df_' + language + '_' + '_tau_start_' + str(
        tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    print('')
    print('')
    print("distance_wins_df is:")
    print(distance_wins_df)
    print("distance_wins_df.shape is:")
    print(distance_wins_df.shape)


    plot_which_model_wins(distance_wins_df, tau_start_for_comparison)
