import pandas as pd
import numpy as np
from scipy.stats import binom, multinomial


# LOAD IN DATA: #
data_exp_1_two_system = pd.read_csv('data/with_counts/TwoSystem.csv', index_col=0)
data_exp_1_three_system = pd.read_csv('data/with_counts/ThreeSystem.csv', index_col=0)


# LOAD IN MODEL PREDICTIONS: #
model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_tau_start_0.1_tau_stop_0.61_tau_step_0.05.csv')


# PARAMETER SETTINGS: #
model = "distance"  # can be set to either "distance" or "person"
language = "Italian"  # can be set to "English", "Italian", "Portuguese" or "Spanish"
object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions
listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
tau_start = 0.1
tau_stop = 0.61
tau_step = 0.05

if language == "English" or language == "Italian":
    data_pd = data_exp_1_two_system
elif language == "Portuguese" or language == "Spanish":
    data_pd = data_exp_1_three_system


# FUNCTION DEFINITIONS: #

def calc_multinom_pmf(pd_model_predictions, pd_data, model, language, object_pos, listener_pos, speaker_tau, listener_tau):
    if language == "English" or language == "Italian":
        WordNo = 2
        words = ["este", "aquel"]
    elif language == "Portuguese" or language == "Spanish":
        WordNo = 3
        words = ["este", "ese", "aquel"]
    probs_per_word = np.zeros((WordNo))
    counts_per_word = np.zeros((WordNo))
    for i in range(len(words)):
        word = words[i]
        model_prediction_row = pd_model_predictions[pd_model_predictions["Model"]==model][pd_model_predictions["Word"]==word][pd_model_predictions["Referent"]==object_pos][pd_model_predictions["Listener_pos"]==listener_pos][pd_model_predictions["WordNo"]==WordNo][pd_model_predictions["SpeakerTau"]==speaker_tau][pd_model_predictions["ListenerTau"]==listener_tau]
        model_prediction_prob = model_prediction_row["Probability"]
        probs_per_word[i] = model_prediction_prob
        # Below is object_pos+1 and listener_pos+1, because in the model_predictions dataframe it starts counting
        # from 0, but in the experimental data dataframe it starts counting from 1.
        data_count_row = pd_data[pd_data["Object_Position"] == object_pos+1][pd_data["Listener_Position"] == listener_pos+1][pd_data["Language"] == language]
        data_count = data_count_row[word]
        counts_per_word[i] = data_count
        total = data_count_row["Total"]
    multinom_pmf = multinomial.pmf(counts_per_word, n=total, p=probs_per_word)
    multinom_logpmf = multinomial.logpmf(counts_per_word, n=total, p=probs_per_word)
    return multinom_pmf[0], multinom_logpmf[0]


def product_logpmf_over_situations(pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau, object_positions, listener_positions):
    log_product = np.log(1.0)  # The first probability should be multiplied with 1.0, which is equivalent to 0.0 in log-space
    prob_product = 1.0
    for object_pos in object_positions:
        for listener_pos in listener_positions:
            multinom_pmf, multinom_logpmf = calc_multinom_pmf(pd_model_predictions, pd_data, model, language, object_pos, listener_pos, speaker_tau, listener_tau)
            log_product += multinom_logpmf  # multiplication in probability space is equivalent to addition in log-space
            prob_product *= multinom_pmf
    return log_product, prob_product


def likelihood_across_parameter_settings(pd_model_predictions, pd_data, model, language, tau_start, tau_stop, tau_step, object_positions, listener_positions):
    log_likelihood_dict = {"SpeakerTau":[],
                   "ListenerTau":[],
                   "LogLikelihood":[]}
    likelihood_dict = {"SpeakerTau":[],
                   "ListenerTau":[],
                   "Likelihood":[]}
    for listener_rationality in np.arange(tau_start, tau_stop, tau_step):
        print('')
        print(f"listener_rationality is {listener_rationality}:")
        for speaker_rationality in np.arange(tau_start, tau_stop, tau_step):
            print(f"speaker_rationality is {speaker_rationality}:")
            log_product, prob_product = product_logpmf_over_situations(pd_model_predictions, pd_data, model, language, round(speaker_rationality, 2), round(listener_rationality, 2), object_positions, listener_positions)
            log_likelihood_dict["SpeakerTau"].append(speaker_rationality)
            log_likelihood_dict["ListenerTau"].append(listener_rationality)
            log_likelihood_dict["LogLikelihood"].append(log_product)
            likelihood_dict["SpeakerTau"].append(speaker_rationality)
            likelihood_dict["ListenerTau"].append(listener_rationality)
            likelihood_dict["Likelihood"].append(prob_product)
    log_likelihood_df = pd.DataFrame(data=log_likelihood_dict)
    likelihood_df = pd.DataFrame(data=likelihood_dict)
    return log_likelihood_df, likelihood_df


print('')
print('')
print(f"LANGUAGE = {language} + MODEL = {model}:")
log_likelihood_df, likelihood_df = likelihood_across_parameter_settings(model_predictions, data_pd, model, language, tau_start, tau_stop, tau_step, object_positions, listener_positions)
print('')
print('')
print("log_likelihood_df is:")
print(log_likelihood_df)
print('')
print('')
print("likelihood_df is:")
print(likelihood_df)


log_likelihood_df.to_pickle('./log_likelihood_df_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')

likelihood_df.to_pickle('./likelihood_df_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')