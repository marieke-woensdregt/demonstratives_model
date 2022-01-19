import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
experiment = "attention"  # Can be set to either "baseline" (=Experiment 1) or "attention" (=Experiment 2)

if experiment == "attention":
    models = ["distance_attention", "person_attention"]
else:
    models = ["distance", "person"]
# languages = ["English", "Italian", "Portuguese", "Spanish"]
if experiment == "attention":
    object_positions = [1, 2, 3]  # array of all possible object (= referent) positions
else:
    object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions

listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
listener_attentions = [0, 1, 2, 3]  # array of all possible listener positions

tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

transparent_plots = False  # Can be set to True or False

language_combos = [["English", "Italian"], ["Portuguese", "Spanish"]]

best_fit_parameters_exp1_dict = {"English":[0.75, 0.95],
                                 "Italian":[0.45, 1.2],
                                 "Portuguese":[0.45, 0.9],
                                 "Spanish":[0.45, 0.9]}

best_fit_parameters_exp2_dict = {"English":[1.7, 1.15],
                                 "Italian":[0.65, 1.],
                                 "Portuguese":[0.55, 1.65],
                                 "Spanish":[0.5, 1.95]}


def create_probs_and_proportions_dataframe(experiment, pd_model_predictions, pd_data, model, language_combo, speaker_tau_per_language, listener_tau_per_language, object_positions, listener_positions, listener_attentions):

    if language_combo == ["English", "Italian"]:
        WordNo = 2
        words = ["este", "aquel"]
    elif language_combo == ["Portuguese", "Spanish"]:
        WordNo = 3
        words = ["este", "ese", "aquel"]

    if experiment == "attention":
        merged_row_dict = {"Model":[],
                           "WordNo": [],
                           "SpeakerTau": [],
                           "ListenerTau": [],
                           "Language":[],
                           "Referent":[],
                           "Listener_att":[],
                           "Word":[],
                           "Probability_model":[],
                           "Proportion_data":[]}
    else:
        merged_row_dict = {"Model":[],
                           "WordNo": [],
                           "SpeakerTau": [],
                           "ListenerTau": [],
                           "Language":[],
                           "Referent":[],
                           "Listener_pos":[],
                           "Word":[],
                           "Probability_model":[],
                           "Proportion_data":[]}

    if experiment == "attention":
        for i in range(len(language_combo)):
            language = language_combo[i]
            speaker_tau = speaker_tau_per_language[i]
            listener_tau = listener_tau_per_language[i]

            for object_pos in object_positions:
                for listener_att in listener_attentions:
                    for word in words:

                        model_prediction_row = pd_model_predictions[pd_model_predictions["Model"]==model][pd_model_predictions["Word"]==word][pd_model_predictions["Referent"]==object_pos][pd_model_predictions["Listener_att"]==listener_att][pd_model_predictions["WordNo"]==WordNo][pd_model_predictions["SpeakerTau"]==speaker_tau][pd_model_predictions["ListenerTau"]==listener_tau]

                        # Below is object_pos+1 and listener_pos+1, because in the model_predictions dataframe it starts counting
                        # from 0, but in the experimental data dataframe it starts counting from 1.
                        # TODO: Currently the data dataframe looks different for Experiment 1 and Experiment 2, where the relevant columns in the dataframe for Experiment are called things like "Estep" and "Aquelp", whereas for Experiment 2, the dataframe contains seperate rows for the separate words (e.g. an "este" row and an "aquel" row), and the relevant column within that row is called "Percentage". That's why below the specification "[pd_data["Word"] == word]" is added, whereas that isn't present under the condition where experiment == "baseline" (i.e. for Experiment 1).
                        data_count_row = pd_data[pd_data["Object_Position"] == object_pos+1][pd_data["Listener_Attention"] == listener_att+1][pd_data["Language"] == language][pd_data["Word"] == word]

                        pd.set_option('display.max_columns', None)

                        merged_row_dict["Model"].append(model)
                        merged_row_dict["WordNo"].append(WordNo)
                        merged_row_dict["SpeakerTau"].append(speaker_tau)
                        merged_row_dict["ListenerTau"].append(listener_tau)
                        merged_row_dict["Language"].append(language)
                        merged_row_dict["Referent"].append(object_pos)
                        merged_row_dict["Listener_att"].append(listener_att)
                        merged_row_dict["Word"].append(word)
                        merged_row_dict["Probability_model"].append(float(model_prediction_row["Probability"]))
                        # TODO: Currently the data dataframe looks different for Experiment 1 and Experiment 2, where the relevant columns in the dataframe for Experiment are called things like "Estep" and "Aquelp", whereas for Experiment 2, the dataframe contains seperate rows for the separate words (e.g. an "este" row and an "aquel" row), and the relevant column within that row is called "Percentage"
                        merged_row_dict["Proportion_data"].append(float(data_count_row["Percentage"]))

    else:
        for i in range(len(language_combo)):
            language = language_combo[i]
            speaker_tau = speaker_tau_per_language[i]
            listener_tau = listener_tau_per_language[i]
            for object_pos in object_positions:
                for listener_pos in listener_positions:
                    for word in words:

                        model_prediction_row = pd_model_predictions[pd_model_predictions["Model"]==model][pd_model_predictions["Word"]==word][pd_model_predictions["Referent"]==object_pos][pd_model_predictions["Listener_pos"]==listener_pos][pd_model_predictions["WordNo"]==WordNo][pd_model_predictions["SpeakerTau"]==speaker_tau][pd_model_predictions["ListenerTau"]==listener_tau]

                        # Below is object_pos+1 and listener_pos+1, because in the model_predictions dataframe it starts counting
                        # from 0, but in the experimental data dataframe it starts counting from 1.
                        # TODO: Currently the data dataframe looks different for Experiment 1 and Experiment 2, where the relevant columns in the dataframe for Experiment are called things like "Estep" and "Aquelp", whereas for Experiment 2, the dataframe contains seperate rows for the separate words (e.g. an "este" row and an "aquel" row), and the relevant column within that row is called "Percentage". That's why below the specification "[pd_data["Word"] == word]" is not present, whereas for the condition where experiment == "attention" (i.e. for Experiment 2), it is.
                        data_count_row = pd_data[pd_data["Object_Position"] == object_pos+1][pd_data["Listener_Position"] == listener_pos+1][pd_data["Language"] == language]

                        pd.set_option('display.max_columns', None)

                        merged_row_dict["Model"].append(model)
                        merged_row_dict["WordNo"].append(WordNo)
                        merged_row_dict["SpeakerTau"].append(speaker_tau)
                        merged_row_dict["ListenerTau"].append(listener_tau)
                        merged_row_dict["Language"].append(language)
                        merged_row_dict["Referent"].append(object_pos)
                        merged_row_dict["Listener_pos"].append(listener_pos)
                        merged_row_dict["Word"].append(word)
                        merged_row_dict["Probability_model"].append(float(model_prediction_row["Probability"]))
                        # TODO: Currently the data dataframe looks different for Experiment 1 and Experiment 2, where the relevant columns in the dataframe for Experiment are called things like "Estep" and "Aquelp", whereas for Experiment 2, the dataframe contains seperate rows for the separate words (e.g. an "este" row and an "aquel" row), and the relevant column within that row is called "Percentage"
                        merged_row_dict["Proportion_data"].append(float(data_count_row[word.capitalize()+"p"]))

    pd_probs_and_proportions_over_trials = pd.DataFrame.from_dict(merged_row_dict)

    return pd_probs_and_proportions_over_trials




def plot_scatter_model_against_data(pd_probs_and_proportions_over_trials, experiment, model, language_combo, transparent_plots):
    # set seaborn plotting aesthetics
    if transparent_plots is True:
        sns.set(style='white')
    else:
        sns.set(style='whitegrid')
    sns.set_palette("colorblind")

    sns.scatterplot(data=pd_probs_and_proportions_over_trials, x="Probability_model", y="Proportion_data", hue="Language")

    if experiment == "attention":
        plt.title(f"Correlation {model.capitalize()} Model * Experiment 2", fontsize=17)
    else:
        plt.title(f"Correlation {model.capitalize()} Model * Experiment 1", fontsize=17)
    if transparent_plots is True:
        plt.savefig('plots/'+'scatter_'+experiment+'_'+str(language_combo).replace(" ", "")+'_'+model+'.png', transparent=transparent_plots)
    else:
        plt.savefig('plots/'+'scatter_'+experiment+'_'+str(language_combo).replace(" ", "")+'_'+model+'.pdf', transparent=transparent_plots)
    plt.show()



for language_combo in language_combos:
    print('')
    print('')
    print(language_combo)

    # LOAD IN DATA: #
    if experiment == "attention":
        if language_combo == ["English", "Italian"]:
            data_pd = pd.read_csv('data/experiment_2/with_counts/TwoSystem_Attention.csv', index_col=0)
        elif language_combo == ["Portuguese", "Spanish"]:
            data_pd = pd.read_csv('data/experiment_2/with_counts/ThreeSystem_Attention.csv', index_col=0)
    else:
        if language_combo == ["English", "Italian"]:
            data_pd = pd.read_csv('data/experiment_1/with_counts/TwoSystem.csv', index_col=0)
        elif language_combo == ["Portuguese", "Spanish"]:
            data_pd = pd.read_csv('data/experiment_1/with_counts/ThreeSystem.csv', index_col=0)

    if experiment == "attention":
        best_fit_parameters = best_fit_parameters_exp2_dict
    else:
        best_fit_parameters = best_fit_parameters_exp1_dict

    for model in models:

        speaker_tau_per_language = []
        listener_tau_per_language = []
        for language in language_combo:
            speaker_tau = best_fit_parameters[language][0]
            listener_tau = best_fit_parameters[language][1]
            speaker_tau_per_language.append(speaker_tau)
            listener_tau_per_language.append(listener_tau)


        print('')
        print('')
        print("speaker_tau_per_language is:")
        print(speaker_tau_per_language)
        print('')
        print("listener_tau_per_language is:")
        print(listener_tau_per_language)

        # LOAD IN MODEL PREDICTIONS: #
        if experiment == "attention":

            if "attention" in model: #TODO: Get rid of this ad-hoc solution and make it more organised
                models_for_filename = ["distance_attention", "person_attention"]
                model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

            else:
                model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
        else:
            model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_' +str(models).replace(" ", "")+'_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')


        pd_probs_and_proportions_over_trials = create_probs_and_proportions_dataframe(experiment, model_predictions, data_pd, model, language_combo, speaker_tau_per_language, listener_tau_per_language, object_positions, listener_positions, listener_attentions)

        pd.set_option('display.max_columns', None)
        print('')
        print('')
        print("pd_probs_and_proportions_over_trials is:")
        print(pd_probs_and_proportions_over_trials)



        plot_scatter_model_against_data(pd_probs_and_proportions_over_trials, experiment, model, language_combo, transparent_plots)