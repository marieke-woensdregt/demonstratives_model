import pandas as pd
import numpy as np
from plot_scatter_plots_model_against_data import pearsonr_ci
import matplotlib.pyplot as plt
import seaborn as sns


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

listener_attentions = [0, 1, 2, 3, 4]
object_positions = [1, 2, 3, 4]


tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

best_fitting = False  # Can be set to either True (will focus only on best-fitting parameter settings) or False (will do # analysis across whole parameter range)

only_misaligned = True  # Can be set to either True (will only analyse misaligned trials) or False (will analyse all trials)


if len(listener_attentions) == 4:
    sigma_dict_across_parameters = {"English": 0.19,
                  "Italian": 0.05,
                  "Portuguese": 0.18,
                  "Spanish": 0.13}  # Variance values for Gaussian used to add noise to the model predictions

    sigma_dict_best_fitting = {"English": 0.12,
                  "Italian": 0.07,
                  "Portuguese": 0.24,
                  "Spanish": 0.2}  # Variance values for Gaussian used to add noise to the model predictions

elif len(listener_attentions) == 5:
    sigma_dict_across_parameters = {"English": 0.23,
                  "Italian": 0.06,
                  "Portuguese": 0.22,
                  "Spanish": 0.16}  # Variance values for Gaussian used to add noise to the model predictions

    sigma_dict_across_parameters_misaligned_only = {"English": 0.23,
                  "Italian": 0.06,
                  "Portuguese": 0.22,
                  "Spanish": 0.16}  # Variance values for Gaussian used to add noise to the model predictions

    sigma_dict_best_fitting = {"English": 0.11,
                  "Italian": 0.08,
                  "Portuguese": 0.28,
                  "Spanish": 0.22}  # Variance values for Gaussian used to add noise to the model predictions

    sigma_dict_best_fitting_misaligned_only = {"English": 0.12,
                  "Italian": 0.09,
                  "Portuguese": 0.25,
                  "Spanish": 0.22}  # Variance values for Gaussian used to add noise to the model predictions



best_fit_parameters_baseline_exp2_dict = {"English":[1.3, 1.75],
                                 "Italian":[0.5, 1.5],
                                 "Portuguese":[0.65, 1.65],
                                 "Spanish":[0.65, 1.65]}


best_fit_parameters_attention_exp2_dict = {"English":[1.7, 1.15],
                                 "Italian":[0.65, 1.],
                                 "Portuguese":[0.55, 1.65],
                                 "Spanish":[0.5, 1.95]}



#
# best_fit_parameters_baseline_exp2_dict = {"English":[1.3, 1.8],
#                                  "Italian":[0.5, 1.5],
#                                  "Portuguese":[0.7, 1.7],
#                                  "Spanish":[0.7, 1.7]}
#
#
# best_fit_parameters_attention_exp2_dict = {"English":[1.7, 1.2],
#                                  "Italian":[0.7, 1.],
#                                  "Portuguese":[0.6, 1.7],
#                                  "Spanish":[0.5, 2.0]}



transparent_plots = False  # Can be set to True or False


# FUNCTIONS DEFINITIONS: #

def plot_scatter_model_against_data(pd_probs_and_proportions_over_trials, experiment, model, transparent_plots):
    # set seaborn plotting aesthetics
    if transparent_plots is True:
        sns.set(style='white')
    else:
        sns.set(style='whitegrid')
    sns.set_palette("colorblind")

    sns.scatterplot(data=pd_probs_and_proportions_over_trials, x="Probability", y="Noisy_probs")

    if experiment == "attention":
        if "attention" in model:
            plt.title(f"{language.capitalize()}: Attention Model vs. Noisy Attention Model", fontsize=17)
        else:
            plt.title(f"{language.capitalize()}: Baseline Model vs. Noisy Attention Model", fontsize=17)
    else:
        plt.title(f"{language.capitalize()}, {model.capitalize()}, Exp. 1, Model * Model", fontsize=17)
    if transparent_plots is True:
        plt.savefig('plots/'+'scatter_model_vs_noisy_model_best_fit_'+str(best_fitting)+'_misaligned_only_'+str(only_misaligned)+'_'+experiment+'_n_positions_'+str(len(listener_attentions)) +'_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) +'_'+language+'_'+model+'.png', transparent=transparent_plots)
    else:
        plt.savefig('plots/'+'scatter_model_vs_noisy_model_best_fit_'+str(best_fitting)+'_misaligned_only_'+str(only_misaligned)+'_'+experiment+'_n_positions_'+str(len(listener_attentions)) +'_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) +'_'+language+'_'+model+'.pdf', transparent=transparent_plots)
    plt.show()


for language in languages:
    if language == "English" or language == "Italian":
        models = ['distance', 'distance_attention']
    elif language == "Portuguese" or language == "Spanish":
        models = ['person', 'person_attention']


    if best_fitting is True:
        if only_misaligned is True:
            sigma_dict = sigma_dict_best_fitting_misaligned_only
        else:
            sigma_dict = sigma_dict_best_fitting
    else:
        if only_misaligned is True:
            sigma_dict = sigma_dict_across_parameters_misaligned_only
        else:
            sigma_dict = sigma_dict_across_parameters

    sigma = sigma_dict[language]
    print("sigma is:")
    print(sigma)

    # LOAD IN MODEL PREDICTIONS: #

    model_predictions_dict = {}

    for model in models:

        print('')
        print('')
        print(f"LANGUAGE = {language} + MODEL = {model}:")

        if rsa_layer is True:

            if experiment == "attention":
                if "attention" in model: #TODO: Get rid of this ad-hoc solution and make it more organised
                    # models_for_filename = ["distance_attention", "person_attention"]
                    models_for_filename = [model]
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_n_positions_'+str(len(listener_attentions)) + str(models_for_filename).replace(" ", "") + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')
                else:
                    # models_for_filename = ["distance", "person"]
                    models_for_filename = [model]
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_n_positions_'+str(len(listener_attentions)) + str(models_for_filename).replace(" ", "") + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

            else:
                model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_n_positions_'+str(len(listener_attentions)) + str(models).replace(" ", "") + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

        else:
            if experiment == "attention":
                if "attention" in model: #TODO: Get rid of this ad-hoc solution and make it more organised
                    models_for_filename = ["distance_attention", "person_attention"]
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_Simple_Attention_Ese_uniform_' + str(ese_uniform) + '_' + str(models_for_filename).replace(" ", "") + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

                else:
                    models_for_filename = ["distance", "person"]
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_Simple_Attention_Ese_uniform_' + str(ese_uniform) + '_' + str(models_for_filename).replace(" ", "") + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

            else:
                model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_Simple_Ese_uniform_' + str(ese_uniform) + '_' + str(models).replace(" ", "") + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')


        if only_misaligned is True:
            aligned_column = np.where(model_predictions["Referent"] == model_predictions["Listener_att"], True, False)
            model_predictions["Aligned"] = aligned_column
            model_predictions = model_predictions[model_predictions["Aligned"] == False]
            # pd.set_option('display.max_columns', None)
            # print('')
            # print('')
            # print("model_predictions ONLY MISALIGNED TRIALS are:")
            # print(model_predictions)


        if best_fitting is True:
            if "attention" in model:
                speaker_tau = best_fit_parameters_attention_exp2_dict[language][0]
                listener_tau = best_fit_parameters_attention_exp2_dict[language][1]
            else:
                speaker_tau = best_fit_parameters_baseline_exp2_dict[language][0]
                listener_tau = best_fit_parameters_baseline_exp2_dict[language][1]
            print('')
            print('')
            print("speaker_tau is:")
            print(speaker_tau)
            print("listener_tau is:")
            print(listener_tau)

            pd.set_option('display.max_columns', None)
            # print("model_predictions are:")
            # print(model_predictions)

            # print('')
            # print('')
            # print("model_predictions BEFORE HONING IN ON BEST-FITTING PARAMETERS:")
            # print(model_predictions)

            model_predictions = model_predictions[model_predictions["SpeakerTau"] == speaker_tau][model_predictions["ListenerTau"] == listener_tau]

            # print('')
            # print('')
            # print("model_predictions AFTER HONING IN ON BEST-FITTING PARAMETERS:")
            # print(model_predictions)


        if "attention" in model:
            model_predictions_dict["Attention"] = model_predictions
        else:
            model_predictions_dict["Baseline"] = model_predictions


    # ADD NOISE TO ATTENTION MODEL PREDICTIONS: #

    attention_model_predictions = model_predictions_dict["Attention"]
    # print('')
    # print("attention_model_predictions are:")
    # print(attention_model_predictions)

    prob_column = attention_model_predictions["Probability"]
    # print("prob_column is:")
    # print(prob_column)

    prob_column_array = prob_column.to_numpy()
    # print("prob_column_array is:")
    # print(prob_column_array)

    noisy_probs_array = np.random.normal(prob_column_array, sigma)
    # print("noisy_probs are:")
    # print(noisy_probs_array)


    # ADD NOISY PREDICTIONS TO BOTH BASELINE AND ATTENTION DATAFRAME: #

    for model in models:
        print('')
        print('')
        print("model is:")
        print(model)

        if "attention" in model:
            model_predictions = model_predictions_dict["Attention"]
        else:
            model_predictions = model_predictions_dict["Baseline"]

        model_predictions["Noisy_probs"] = noisy_probs_array
        # print("model_predictions are:")
        # print(model_predictions)

        model_probs = model_predictions["Probability"][model_predictions["Model"] == model]
        # print("model_probs are:")
        # print(model_probs)

        noisy_probs = model_predictions["Noisy_probs"][model_predictions["Model"] == model]
        # print("noisy_probs are:")
        # print(noisy_probs)


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


        # MAKE SCATTER PLOT: #
        plot_scatter_model_against_data(model_predictions, experiment, model, transparent_plots)
