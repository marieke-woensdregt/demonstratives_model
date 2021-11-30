import pandas as pd
import numpy as np
from scipy.stats import binom, multinomial

# PARAMETER SETTINGS: #
models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]
object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions
listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
tau_start = 0.4
tau_stop = 4.1
tau_step = 0.1


experiment = "attention"


# FUNCTION DEFINITIONS: #

def calc_multinom_pmf(pd_model_predictions, pd_data, model, language, object_pos, listener_pos, speaker_tau, listener_tau):
    """
    Calculates the likelihood and log_likelihood of the parameters speaker_tau and listener_tau given the data.

    :param pd_model_predictions: Pandas dataframe containing the model predictions. Should contain at least the following columns: "Model", "Word", "Referent", "Listener_pos", "WordNo", "SpeakerTau", and "ListenerTau".
    :param pd_data: Pandas dataframe containing the experimental data. Should contain at least the following columns: "Object_Position", "Listener_Position", "Language"
    :param model: String; Model to be used to produce model predictions: "distance" or "person"
    :param language: String; Language from which the experimental data should be taken: "English", "Italian", "Portuguese" or "Spanish"
    :param object_pos: Integer; Object position (i.e. position of the referent) that should be looked at
    :param listener_pos: Integer; Listener position that should be looked at
    :param speaker_tau: Float; Speaker rationality value that should be used to calculate likelihood
    :param listener_tau: Float; Listener rationality value that should be used to calculate likelihood
    :return: (i) Float, (ii) float; (i) LOG likelihood of parameters given data, (ii) Likelihood of parameters given data
    """
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
    multinom_logpmf = multinomial.logpmf(counts_per_word, n=total, p=probs_per_word)
    multinom_pmf = multinomial.pmf(counts_per_word, n=total, p=probs_per_word)
    return multinom_logpmf[0], multinom_pmf[0]


def product_logpmf_over_situations(pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau, object_positions, listener_positions):
    """
    Calculates the product of the likelihood values (i.e. sum of the LOG likelihood values) across all situations (i.e. all different trials) in the experiment. That is: all different combinations of possible object_positions and listener_positions.

    :param pd_model_predictions: Pandas dataframe containing the model predictions. Should contain at least the following columns: "Model", "Word", "Referent", "Listener_pos", "WordNo", "SpeakerTau", and "ListenerTau".
    :param pd_data: Pandas dataframe containing the experimental data. Should contain at least the following columns: "Object_Position", "Listener_Position", "Language"
    :param model: String; Model to be used to produce model predictions: "distance" or "person"
    :param language: String; Language from which the experimental data should be taken: "English", "Italian", "Portuguese" or "Spanish"
    :param speaker_tau: Float; Speaker rationality value that should be used to calculate likelihood
    :param listener_tau: Float; Listener rationality value that should be used to calculate likelihood
    :param object_positions: List of integers; All possible object (i.e. referent) positions
    :param listener_positions: List of integers; All possible listener positions
    :return: (i) Float; (ii) float; (i) The sum of all LOG likelihoods (because multiplication in non-log space is equal to addition in log-space), (ii) the product of all likelihoods
    """
    log_product = np.log(1.0)  # The first probability should be multiplied with 1.0, which is equivalent to 0.0 in log-space
    product = 1.0
    for object_pos in object_positions:
        for listener_pos in listener_positions:
            multinom_logpmf, multinom_pmf = calc_multinom_pmf(pd_model_predictions, pd_data, model, language, object_pos, listener_pos, speaker_tau, listener_tau)
            #TODO: Should I be using logaddexp() or logsumexp() instead?
            log_product += multinom_logpmf  # multiplication in probability space is equivalent to addition in log-space
            product *= multinom_pmf
    return log_product, product


def likelihood_across_parameter_settings(pd_model_predictions, pd_data, model, language, tau_start, tau_stop, tau_step, object_positions, listener_positions):
    """
    Calculates the sum of LOG likelihoods and product of likelihoods of the parameters given the data across a range of combinations of possible parameter settings for speaker_rationality (i.e. SpeakerTau) and listener_rationality (i.e. ListenerTau).

    :param pd_model_predictions: Pandas dataframe containing the model predictions. Should contain at least the following columns: "Model", "Word", "Referent", "Listener_pos", "WordNo", "SpeakerTau", and "ListenerTau". The columns "SpeakerTau" and "ListenerTau" should contain the full range of parameter settings corresponding to the input arguments tau_start, tau_stop, and tau_step.
    :param pd_data: Pandas dataframe containing the experimental data. Should contain at least the following columns: "Object_Position", "Listener_Position", "Language"
    :param model: String; Model to be used to produce model predictions: "distance" or "person"
    :param language: String; Language from which the experimental data should be taken: "English", "Italian", "Portuguese" or "Spanish"
    :param tau_start: Float; Starting point of range of parameter settings that should be explored
    :param tau_stop: Float; End point of range of parameter settings that should be explored
    :param tau_step: Float; Step size for range of parameter settings that should be explored
    :param object_positions: List of integers; All possible object (i.e. referent) positions
    :param listener_positions: List of integers; All possible listener positions
    :return: (i) Pandas dataframe, (ii) Pandas dataframe; (i) dataframe containing LOG likelihood values, with the following three columns: "SpeakerTau", "ListenerTau", and "LogLikelihood", (ii) dataframe containing likelihood values, with the following three columns: "SpeakerTau", "ListenerTau", and "Likelihood",
    """
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


# LOAD IN MODEL PREDICTIONS: #
if experiment == "attention":
    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
else:
    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(
        tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

for language in languages:
    print('')
    print('')
    print(language)

    # LOAD IN DATA: #
    if experiment == "attention":

        if language == "English" or language == "Italian":
            data_pd = pd.read_csv('data/with_counts/TwoSystem.csv', index_col=0)
        elif language == "Portuguese" or language == "Spanish":
            data_pd = pd.read_csv('data/with_counts/ThreeSystem.csv', index_col=0)

    else:
        if language == "English" or language == "Italian":
            data_pd = pd.read_csv('data/with_counts/TwoSystem.csv', index_col=0)
        elif language == "Portuguese" or language == "Spanish":
            data_pd = pd.read_csv('data/with_counts/ThreeSystem.csv', index_col=0)

    for model in models:
        print('')
        print(model)

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

        if experiment == "attention":
            log_likelihood_df.to_pickle('model_fitting_data/'+'log_likelihood_df_Attention_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')

            likelihood_df.to_pickle('model_fitting_data/'+'likelihood_df_Attention_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')
        else:
            log_likelihood_df.to_pickle('model_fitting_data/'+'log_likelihood_df_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')

            likelihood_df.to_pickle('model_fitting_data/'+'likelihood_df_'+language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')