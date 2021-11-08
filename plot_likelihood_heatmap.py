import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
model = "distance"  # can be set to either "distance" or "person"
language = "Spanish"  # can be set to "English", "Italian", "Portuguese" or "Spanish"
tau_start = 0.1
tau_stop = 0.61
tau_step = 0.05

likelihood_df = pd.read_pickle('./log_likelihood_df_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')
print("likelihood_df is:")
print(likelihood_df)

def plot_heatmap(likelihood_df):
    fig, ax = plt.subplots(figsize=(11, 9))
    likelihood_df = likelihood_df.pivot(index='SpeakerTau', columns='ListenerTau', values='LogLikelihood')
    print("likelihood_df AFTER PIVOT is:")
    print(likelihood_df)
    sns.heatmap(likelihood_df)
    ax = sns.heatmap(likelihood_df,
                     xticklabels=likelihood_df.columns.values.round(2),
                     yticklabels=likelihood_df.index.values.round(2))
    plt.title(f"{language} + {model}")
    plt.savefig('heatmap_log_likelihood_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()

plot_heatmap(likelihood_df)