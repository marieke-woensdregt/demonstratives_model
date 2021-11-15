import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
# model = "person"  # can be set to either "distance" or "person"
# language = "Spanish"  # can be set to "English", "Italian", "Portuguese" or "Spanish"
tau_start = 0.41
tau_stop = 1.41
tau_step = 0.02


def plot_heatmap(likelihood_df):
    """
    Plots the (log) likelihoods of the parameters given the data in a 2D heatmap with speaker_rationality ("SpeakerTau") on the y-axis and listener_rationality ("ListenerTau") on the x-axis.

    :param likelihood_df: Pandas dataframe containing (log) likelihoods. Should contain the following three columns: 'SpeakerTau', 'ListenerTau', and 'LogLikelihood'
    :return: Saves the figure in .pdf format, and shows it
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    #TODO: Should I consider replacing all -inf values in the log likelihood df with NAN? Just to make sure the heatmap is being produced on a fair scale?

    likelihood_df = likelihood_df.pivot(index='SpeakerTau', columns='ListenerTau', values='LogLikelihood')
    sns.heatmap(likelihood_df)
    ax = sns.heatmap(likelihood_df,
                     xticklabels=likelihood_df.columns.values.round(2),
                     yticklabels=likelihood_df.index.values.round(2))
    plt.title(f"{language} + {model}")
    plt.savefig('plots/'+'heatmap_log_likelihood_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()


models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]

for language in languages:
    print('')
    print('')
    print(language)
    for model in models:
        print('')
        print(model)

        log_likelihood_df = pd.read_pickle(
            'model_fitting_data/' + 'log_likelihood_df_' + language + '_' + model + '_tau_start_' + str(
                tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        print("LOG likelihood_df is:")
        print(log_likelihood_df)

        plot_heatmap(log_likelihood_df)