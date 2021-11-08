import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
model = "distance"  # can be set to either "distance" or "person"
language = "English"  # can be set to "English", "Italian", "Portuguese" or "Spanish"
tau_start = 0.1
tau_stop = 0.61
tau_step = 0.05

likelihood_df = pd.read_pickle('./likelihood_df_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')

def plot_heatmap(likelihood_df):
    sns.heatmap(likelihood_df)
    plt.show()
