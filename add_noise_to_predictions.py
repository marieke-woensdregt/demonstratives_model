import pandas as pd
import numpy as np
from scipy.stats import binom, multinomial

# PARAMETER SETTINGS: #

rsa_layer = False  # Can be set to either True or False

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


        print("model_predictions are:")
        print(model_predictions)