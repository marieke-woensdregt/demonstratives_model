import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle5 as p
import pickle


# PARAMETER SETTINGS: #
rsa_layer = False  # Can be set to either True or False

ese_uniform = True  # Can be set to either True or False. Determines whether "ese" under the simple distance model is a uniform distribution (if set to True), or rather centred around the medial objects (if set to False)

experiment = "baseline"

if experiment == "attention":
    models = ["distance", "person", "distance_attention", "person_attention"]
else:
    models = ["distance", "person"]
#models = ["distance", "person", "distance_attention", "person_attention"]  # Possibilities are: ["distance", "person", "distance_attention", "person_attention"]
languages = ["English", "Italian", "Portuguese", "Spanish"]  # Possibilities are: ["English", "Italian", "Portuguese", "Spanish"]
tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

tau_start_for_plot = 0.4
# tau_stop_for_plot = 1.41
tau_start_for_comparison = 0.4
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
    if experiment == "attention":
        plt.savefig('plots/'+'heatmap_log_likelihood_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    else:
        plt.savefig('plots/'+'heatmap_log_likelihood_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' +language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()



def plot_bayes_factor_heatmap(bayes_factor_df, tau_start_for_comparison):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(bayes_factor_df,
                     xticklabels=bayes_factor_df.columns.values.round(2),
                     yticklabels=bayes_factor_df.index.values.round(2))
    plt.title(f"Bayes Factors Distance/Person {language}")
    if experiment == "attention":
        plt.savefig('plots/'+'heatmap_bayes_factors_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    else:
        plt.savefig('plots/'+'heatmap_bayes_factors_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' +language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()



def plot_which_model_wins(distance_wins_df, tau_start_for_comparison):
    fig, ax = plt.subplots(figsize=(11, 9))

    if experiment == "attention":
        vmap = {0: 'baseline', 1: 'attention-correction'}
    else:
        vmap = {0:'person', 1:'distance'}
    n = len(vmap)

    cmap = sns.color_palette("colorblind", n)
    print('')
    print('')
    print("cmap is:")
    print(cmap)
    cmap = ["#d95f02", "#7570b3"]
    print("cmap NEW is:")
    print(cmap)

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

    plt.title(f"Most likely model: {language}")
    if experiment == "attention":
        plt.savefig('plots/'+'heatmap_most_likely_model_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    else:
        plt.savefig('plots/'+'heatmap_most_likely_model_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' +language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()



def plot_strength_of_evidence(evidence_strength_df, tau_start_for_comparison):
    fig, ax = plt.subplots(figsize=(11, 9))

    vmap = {0:'no evidence', 1:'anecdotal', 2:'moderate', 3:'strong', 4:'very strong', 5:'extreme'}
    n = len(vmap)

    cmap = sns.color_palette("flare", n)
    ax = sns.heatmap(evidence_strength_df,
                     xticklabels=evidence_strength_df.columns.values.round(2),
                     yticklabels=evidence_strength_df.index.values.round(2),
                     cmap=cmap)

    # Get the colorbar object from the Seaborn heatmap
    colorbar = ax.collections[0].colorbar
    # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(n)])
    colorbar.set_ticklabels(list(vmap.values()))

    plt.title(f"Strength of evidence in favour of likely model: {language}")
    if experiment == "attention":
        plt.savefig('plots/'+'heatmap_strength_of_evidence_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    else:
        plt.savefig('plots/'+'heatmap_strength_of_evidence_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' +language+'_tau_start_'+str(tau_start_for_comparison)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()




min_log_value_two_system = 0
min_log_value_english = 0
min_log_value_italian = 0
for language in ["English", "Italian"]:
    print('')
    print('')
    print(language)
    for model in models:
        print('')
        print('')
        print(model)

        if experiment == "attention":
            with open('model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(
                    tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                log_likelihood_df = p.load(fh)
            # log_likelihood_df = pd.read_pickle(
            #     'model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(
            #         tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            with open('model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(
                    tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                log_likelihood_df = p.load(fh)
        #     log_likelihood_df = pd.read_pickle(
        #         'model_fitting_data/' + 'log_likelihood_df_' + language + '_' + model + '_tau_start_' + str(
        #             tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
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

        if language == "English":
            if min < min_log_value_english:
                min_log_value_english = min
        elif language == "Italian":
            if min < min_log_value_italian:
                min_log_value_italian = min

        if min < min_log_value_two_system:
            min_log_value_two_system = min


min_log_value_english = rounddown(min_log_value_english)
print('')
print("min_log_value_english is:")
print(min_log_value_english)

min_log_value_italian = rounddown(min_log_value_italian)
print('')
print("min_log_value_italian is:")
print(min_log_value_italian)

min_log_value_two_system = rounddown(min_log_value_two_system)
print('')
print("min_log_value_two_system is:")
print(min_log_value_two_system)


min_log_value_three_system = 0
for language in ["Portuguese", "Spanish"]:
    print('')
    print('')
    print(language)
    for model in models:
        print('')
        print('')
        print(model)

        if experiment == "attention":
            with open('model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                log_likelihood_df = p.load(fh)
            # log_likelihood_df = pd.read_pickle(
            # 'model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(
            #     tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            with open('model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                log_likelihood_df = p.load(fh)
            # log_likelihood_df = pd.read_pickle(
            # 'model_fitting_data/' + 'log_likelihood_df_' + language + '_' + model + '_tau_start_' + str(
            #     tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
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

        if experiment == "attention":
            with open('model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + model + '_tau_start_' + str(
                tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                log_likelihood_df = p.load(fh)
            # log_likelihood_df = pd.read_pickle(
            # 'model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + model + '_tau_start_' + str(
            #     tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            with open('model_fitting_data/' + 'log_likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                log_likelihood_df = p.load(fh)
            # log_likelihood_df = pd.read_pickle(
            # 'model_fitting_data/' + 'log_likelihood_df_' + language + '_' + model + '_tau_start_' + str(
            #     tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        print("LOG likelihood_df is:")
        print(log_likelihood_df)

        if language == "English" or language == "Italian":
            if experiment == "attention":
                if language == "English":
                    plot_likelihood_heatmap(log_likelihood_df, tau_start_for_plot, min_log_value_english)
                elif language == "Italian":
                    plot_likelihood_heatmap(log_likelihood_df, tau_start_for_plot, min_log_value_italian)
            else:
                plot_likelihood_heatmap(log_likelihood_df, tau_start_for_plot, min_log_value_two_system)
        elif language == "Portuguese" or language == "Spanish":
            plot_likelihood_heatmap(log_likelihood_df, tau_start_for_plot, min_log_value_three_system)

        if experiment == "attention":
            with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + model + '_tau_start_' + str(
                tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df = p.load(fh)
            # likelihood_df = pd.read_pickle(
            # 'model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + model + '_tau_start_' + str(
            #     tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df = p.load(fh)
            # likelihood_df = pd.read_pickle(
            # 'model_fitting_data/' + 'likelihood_df_' + language + '_' + model + '_tau_start_' + str(
            #     tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        print("likelihood_df is:")
        print(likelihood_df)


for language in languages:
    print('')
    print('')
    print(language)

    if experiment == "attention":
        bayes_factor_df = pd.read_pickle('model_fitting_data/' + 'bayes_factor_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    else:
        bayes_factor_df = pd.read_pickle('model_fitting_data/' + 'bayes_factor_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

    plot_bayes_factor_heatmap(bayes_factor_df, tau_start_for_comparison)

    if experiment == "attention":
        attention_wins_df = pd.read_pickle('model_fitting_data/' + 'attention_wins_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        # print('')
        # print('')
        # print("attention_wins_df is:")
        # print(attention_wins_df)
        # print("attention_wins_df.shape is:")
        # print(attention_wins_df.shape)

        plot_which_model_wins(attention_wins_df, tau_start_for_comparison)

    else:
        distance_wins_df = pd.read_pickle('model_fitting_data/' + 'distance_wins_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        # print('')
        # print('')
        # print("distance_wins_df is:")
        # print(distance_wins_df)
        # print("distance_wins_df.shape is:")
        # print(distance_wins_df.shape)

        plot_which_model_wins(distance_wins_df, tau_start_for_comparison)

    if experiment == "attention":
        evidence_strength_df = pd.read_pickle('model_fitting_data/' + 'evidence_strength_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    else:
        evidence_strength_df = pd.read_pickle('model_fitting_data/' + 'evidence_strength_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

    print('')
    print('')
    print("evidence_strength_df is:")
    print(evidence_strength_df)
    print("evidence_strength_df.shape is:")
    print(evidence_strength_df.shape)

    plot_strength_of_evidence(evidence_strength_df, tau_start_for_comparison)
