import pandas as pd
import numpy as np
from scipy.stats import binom, multinomial

# PARAMETER SETTINGS: #

rsa_layer = False  # Can be set to either True or False

ese_uniform = True  # Can be set to either True or False. Determines whether "ese" under the simple distance model is a uniform distribution (if set to True), or rather centred around the medial objects (if set to False)

experiment = "attention"
# if experiment == "attention":
#     models = ["distance_attention", "person_attention"]
# else:
#     models = ["distance", "person"]
models = ['distance_attention', 'person_attention']  # ['distance', 'person'] # ['distance_attention', 'person_attention']  # Can contain: 'distance','person','pdhybrid', 'distance_attention', 'person_attention'
languages = ["English", "Italian", "Portuguese", "Spanish"]
#languages = ["Portuguese"]
if experiment == "attention":
    object_positions = [1, 2, 3]  # array of all possible object (= referent) positions
else:
    object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions
listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
listener_attentions = [0, 1, 2, 3]  # array of all possible listener positions
tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05




# FUNCTION DEFINITIONS: #

def calc_multinom_pmf(experiment, pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau, object_pos, listener_pos=None, listener_att=None):
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
    # counts_per_word = []

    for i in range(len(words)):
        word = words[i]
        if experiment == "attention":
            model_prediction_row = pd_model_predictions[pd_model_predictions["Model"]==model][pd_model_predictions["Word"]==word][pd_model_predictions["Referent"]==object_pos][pd_model_predictions["Listener_att"]==listener_att][pd_model_predictions["WordNo"]==WordNo][pd_model_predictions["SpeakerTau"]==speaker_tau][pd_model_predictions["ListenerTau"]==listener_tau]
        else:
            model_prediction_row = pd_model_predictions[pd_model_predictions["Model"]==model][pd_model_predictions["Word"]==word][pd_model_predictions["Referent"]==object_pos][pd_model_predictions["Listener_pos"]==listener_pos][pd_model_predictions["WordNo"]==WordNo][pd_model_predictions["SpeakerTau"]==speaker_tau][pd_model_predictions["ListenerTau"]==listener_tau]
        model_prediction_prob = model_prediction_row["Probability"]
        probs_per_word[i] = model_prediction_prob
        # Below is object_pos+1 and listener_pos+1, because in the model_predictions dataframe it starts counting
        # from 0, but in the experimental data dataframe it starts counting from 1.
        if experiment == "attention":
            data_count_row = pd_data[pd_data["Object_Position"] == object_pos+1][pd_data["Listener_Attention"] == listener_att+1][pd_data["Language"] == language]
            # print('')
            # print("data_count_row is:")
            # print(data_count_row)
        else:
            data_count_row = pd_data[pd_data["Object_Position"] == object_pos+1][pd_data["Listener_Position"] == listener_pos+1][pd_data["Language"] == language]
        if experiment == "attention":
            data_count = data_count_row[word.title()]
            # print('')
            # print("data_count is:")
            # print(data_count)
            # print("len(data_count) is:")
            # print(len(data_count))
        else:
            data_count = data_count_row[word]

        if len(data_count) > 1:
            data_count = np.mean(data_count)  #TODO: Check whether this is really the right way to handle this. For some reason these rows are repeated in the data files (twice for the 2-word languages, and thrice for the 3-word languages), due to my bad R skills for doing data preprocessing...
            # print("data_count AFTER AVERAGING is:")
            # print(data_count)
            counts_per_word[i] = data_count
            # counts_per_word.append(data_count)
        else:
            counts_per_word[i] = data_count
            # counts_per_word.append(data_count)

        # print('')
        # print("counts_per_word is:")
        # print(counts_per_word)
        # print("len(counts_per_word) is:")
        # print(len(counts_per_word))
        # print("len(counts_per_word[0]) is:")
        # print(len(counts_per_word[0]))
        # print("len(counts_per_word[1]) is:")
        # print(len(counts_per_word[1]))

        total = data_count_row["Total"]
    multinom_logpmf = multinomial.logpmf(counts_per_word, n=total, p=probs_per_word)
    multinom_pmf = multinomial.pmf(counts_per_word, n=total, p=probs_per_word)
    return multinom_logpmf[0], multinom_pmf[0]


def product_logpmf_over_situations(experiment, pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau, object_positions, listener_positions=None, listener_attentions=None):
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
    if experiment == "attention":
        for object_pos in object_positions:
            for listener_att in listener_attentions:
                multinom_logpmf, multinom_pmf = calc_multinom_pmf(experiment, pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau, object_pos, listener_att=listener_att)
                #TODO: Should I be using logaddexp() or logsumexp() instead?
                log_product += multinom_logpmf  # multiplication in probability space is equivalent to addition in log-space
                product *= multinom_pmf
    else:
        for object_pos in object_positions:
            for listener_pos in listener_positions:
                multinom_logpmf, multinom_pmf = calc_multinom_pmf(experiment, pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau, object_pos, listener_pos=listener_pos)
                #TODO: Should I be using logaddexp() or logsumexp() instead?
                log_product += multinom_logpmf  # multiplication in probability space is equivalent to addition in log-space
                product *= multinom_pmf
    return log_product, product


def likelihood_across_parameter_settings(experiment, pd_model_predictions, pd_data, model, language, tau_start, tau_stop, tau_step, object_positions, listener_positions=None, listener_attentions=None):
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
            if experiment == "attention":
                log_product, prob_product = product_logpmf_over_situations(experiment, pd_model_predictions, pd_data, model, language, round(speaker_rationality, 2), round(listener_rationality, 2), object_positions, listener_attentions=listener_attentions)
            else:
                log_product, prob_product = product_logpmf_over_situations(experiment, pd_model_predictions, pd_data, model, language, round(speaker_rationality, 2), round(listener_rationality, 2), object_positions, listener_positions=listener_positions)
            log_likelihood_dict["SpeakerTau"].append(speaker_rationality)
            log_likelihood_dict["ListenerTau"].append(listener_rationality)
            log_likelihood_dict["LogLikelihood"].append(log_product)
            likelihood_dict["SpeakerTau"].append(speaker_rationality)
            likelihood_dict["ListenerTau"].append(listener_rationality)
            likelihood_dict["Likelihood"].append(prob_product)
    log_likelihood_df = pd.DataFrame(data=log_likelihood_dict)
    likelihood_df = pd.DataFrame(data=likelihood_dict)
    return log_likelihood_df, likelihood_df



for language in languages:
    print('')
    print('')
    print(language)

    # LOAD IN DATA: #
    if experiment == "attention":
        if language == "English" or language == "Italian":
            data_pd = pd.read_csv('data/experiment_2/with_counts/TwoSystem_Attention.csv', index_col=0)
        elif language == "Portuguese" or language == "Spanish":
            data_pd = pd.read_csv('data/experiment_2/with_counts/ThreeSystem_Attention.csv', index_col=0)
    else:
        if language == "English" or language == "Italian":
            data_pd = pd.read_csv('data/experiment_1/with_counts/TwoSystem.csv', index_col=0)
        elif language == "Portuguese" or language == "Spanish":
            data_pd = pd.read_csv('data/experiment_1/with_counts/ThreeSystem.csv', index_col=0)

    for model in models:
        print('')
        print(model)

        print('')
        print('')
        print(f"LANGUAGE = {language} + MODEL = {model}:")
        #LOAD IN MODEL PREDICTIONS: #

        if rsa_layer is True:

            if experiment == "attention":

                if "attention" in model: #TODO: Get rid of this ad-hoc solution and make it more organised
                    models_for_filename = ["distance_attention", "person_attention"]
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')


                else:
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
            else:
                model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_' +str(models).replace(" ", "")+'_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

        else:
            if experiment == "attention":

                if "attention" in model: #TODO: Get rid of this ad-hoc solution and make it more organised
                    models_for_filename = ["distance_attention", "person_attention"]
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_Simple_Attention_Ese_uniform_' + str(ese_uniform) + '_' + str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

                else:
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_Simple_Attention_Ese_uniform_' + str(ese_uniform) + '_' + str(models).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
            else:
                model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_Simple_Ese_uniform_' + str(ese_uniform) + '_' + str(models).replace(" ", "")+'_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

        if experiment == "attention":
            log_likelihood_df, likelihood_df = likelihood_across_parameter_settings(experiment, model_predictions, data_pd, model, language, tau_start, tau_stop, tau_step, object_positions, listener_attentions=listener_attentions)
        else:
            log_likelihood_df, likelihood_df = likelihood_across_parameter_settings(experiment, model_predictions, data_pd, model, language, tau_start, tau_stop, tau_step, object_positions, listener_positions=listener_positions)
        # print('')
        # print('')
        # print("log_likelihood_df is:")
        # print(log_likelihood_df)
        # print('')
        # print('')
        # print("likelihood_df is:")
        # print(likelihood_df)

        if experiment == "attention":
            log_likelihood_df.to_pickle('model_fitting_data/'+'log_likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')

            likelihood_df.to_pickle('model_fitting_data/'+'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')
        else:
            log_likelihood_df.to_pickle('model_fitting_data/'+'log_likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')

            likelihood_df.to_pickle('model_fitting_data/'+'likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language+'_'+model+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pkl')