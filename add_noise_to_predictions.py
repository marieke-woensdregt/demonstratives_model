import pandas as pd
import numpy as np
from plot_scatter_plots_model_against_data import pearsonr_ci

# PARAMETER SETTINGS: #

rsa_layer = True  # Can be set to either True or False

ese_uniform = True  # Can be set to either True or False. Determines whether "ese" under the simple distance model is a uniform distribution (if set to True), or rather centred around the medial objects (if set to False)

experiment = "attention"  # can be set to either "baseline" or "attention"
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

sigma = 0.14  # Variance value for Gaussian used to add noise to the model predictions


for language in languages:
    if language == "English" or language == "Italian":
        model = 'distance_attention'
    elif language == "Portuguese" or language == "Spanish":
        model = 'person_attention'
    print('')
    print('')
    print(f"LANGUAGE = {language} + MODEL = {model}:")

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

    # LOAD IN MODEL PREDICTIONS: #
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

    # ADD NOISE TO MODEL PREDICTIONS: #
    pd.set_option('display.max_columns', None)
    # print("model_predictions are:")
    # print(model_predictions)

    prob_column = model_predictions["Probability"]
    # print("prob_column is:")
    # print(prob_column)

    prob_column_array = prob_column.to_numpy()
    # print("prob_column_array is:")
    # print(prob_column_array)

    noisy_probs_array = np.random.normal(prob_column_array, sigma)
    # print("noisy_probs are:")
    # print(noisy_probs_array)

    model_predictions_noisy = model_predictions.copy()
    # print("model_predictions_noisy BEFORE ADDING NOISE are:")
    # print(model_predictions_noisy)

    model_predictions_noisy["Probability"] = noisy_probs_array
    # print("model_predictions_noisy AFTER ADDING NOISE are:")
    # print(model_predictions_noisy)

    print('')
    print('')

    model_probs = model_predictions["Probability"][model_predictions["Model"] == model]
    print("model_probs are:")
    print(model_probs)

    noisy_probs = model_predictions_noisy["Probability"][model_predictions_noisy["Model"] == model]
    print("noisy_probs are:")
    print(noisy_probs)

    # CHECK CORRELATIONS BETWEEN ORIGINAL AND NOISY MODEL PREDICTIONS: #
    pearson_correlation = model_probs.corr(noisy_probs)
    print('')
    print("pearson_correlation is:")
    print(pearson_correlation)
    pearson_correlation_reverse = noisy_probs.corr(model_probs)
    print('')
    print("pearson_correlation_reverse is:")
    print(pearson_correlation_reverse)

    r, p, lo, hi = pearsonr_ci(noisy_probs, model_probs, alpha=0.05)
    print('')
    print('')
    print("Pearson's r correlation using code from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/")
    print("r is:")
    print(r)
    print("p is:")
    print(p)
    print("lo is:")
    print(lo)
    print("hi is:")
    print(hi)