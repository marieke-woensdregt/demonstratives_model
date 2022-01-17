import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
languages = ["English", "Italian", "Portuguese", "Spanish"]
baseline_models = ["distance", "person"]
words = [2, 3]
object_positions = [1, 2, 3]  # array of all possible object (= referent) positions
listener_attentions = [0, 1, 2, 3]  # array of all possible listener positions
tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

absolute = False

numbering = "experiment"  # can be set to either "experiment" ([1, 2, 3, 4]) or "python" ([0, 1, 2, 3])


# FUNCTION DEFINITIONS: #


def get_most_contrastive_trials(pd_difference_per_row, numbering, model, WordNo, speaker_rationality=None, listener_rationality=None):
    if WordNo == 2:
        words = ["este", "aquel"]
    elif WordNo == 3:
        words = ["este", "ese", "aquel"]
    data_selection = pd_difference_per_row[pd_difference_per_row["Model"] == model]
    data_selection = data_selection[data_selection["WordNo"] == WordNo]
    if speaker_rationality != None:
        data_selection = data_selection[data_selection["SpeakerTau"] == speaker_rationality]
    if listener_rationality != None:
        data_selection = data_selection[data_selection["ListenerTau"] == listener_rationality]
    most_contrastive_trials_dict = {"Word":[],
                                    "SpeakerTau":[],
                                    "ListenerTau":[],
                                    # "Speaker_pos":[],
                                    # "Listener_pos":[],
                                    "Referent": [],
                                    "Listener_att":[],
                                    "Prob_difference":[]}
    for word in words:
        data_subset_for_word = data_selection[data_selection["Word"] == word]
        for polarity in ["Min", "Max"]:
            if polarity == "Min":
                most_different = data_subset_for_word["Prob_difference"].min()
            elif polarity == "Max":
                most_different = data_subset_for_word["Prob_difference"].max()
            most_different_rows = data_subset_for_word.loc[data_subset_for_word["Prob_difference"] == most_different]
            for index, row in most_different_rows.iterrows():
                most_contrastive_trials_dict["Word"].append(word)
                most_contrastive_trials_dict["SpeakerTau"].append(row["SpeakerTau"])
                most_contrastive_trials_dict["ListenerTau"].append(row["ListenerTau"])
                if numbering == "experiment":
                    # most_contrastive_trials_dict["Speaker_pos"].append(row["Speaker_pos"]+1)
                    # most_contrastive_trials_dict["Listener_pos"].append(row["Listener_pos"]+1)
                    most_contrastive_trials_dict["Listener_att"].append(row["Listener_att"]+1)
                    most_contrastive_trials_dict["Referent"].append(row["Referent"]+1)
                else:
                    # most_contrastive_trials_dict["Speaker_pos"].append(row["Speaker_pos"])
                    # most_contrastive_trials_dict["Listener_pos"].append(row["Listener_pos"])
                    most_contrastive_trials_dict["Listener_att"].append(row["Listener_att"])
                    most_contrastive_trials_dict["Referent"].append(row["Referent"])
                most_contrastive_trials_dict["Prob_difference"].append(row["Prob_difference"])
    pd_most_contrastive_trials = pd.DataFrame.from_dict(most_contrastive_trials_dict)
    return pd_most_contrastive_trials



def contrastive_trials_across_parameters(baseline_models, words):
    for model in baseline_models:
        if absolute is True:
            pd_difference_per_row = pd.read_pickle(
                'model_predictions/' + 'pd_abs_difference_in_model_predictions_' + model + '_tau_start_' + str(
                    tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            pd_difference_per_row = pd.read_pickle(
                'model_predictions/' + 'pd_difference_in_model_predictions_' + model + '_tau_start_' + str(
                    tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

        for WordNo in words:
            print('')
            print('')
            print(f"MODEL = {model}; WORDNO = {WordNo}:")
            pd_most_contrastive_trials = get_most_contrastive_trials(pd_difference_per_row, numbering, model, WordNo)
            pd.set_option('display.max_columns', None)
            print("pd_most_contrastive_trials is:")
            print(pd_most_contrastive_trials)



def contrastive_trials_for_best_parameters(best_fit_parameters, languages):
    for language in languages:
        if language == "English" or language == "Italian":
            model = "distance"
            WordNo = 2
        elif language == "Portuguese" or language == "Spanish":
            model = "person"
            WordNo = 3
        print('')
        print('')
        print(f"LANGUAGE = {language}; MODEL = {model}:")
        speaker_rationality = best_fit_parameters[language][0]
        listener_rationality = best_fit_parameters[language][1]
        print("speaker_rationality is:")
        print(speaker_rationality)
        print("listener_rationality is:")
        print(listener_rationality)

        if absolute is True:
            pd_difference_per_row = pd.read_pickle(
                'model_predictions/' + 'pd_abs_difference_in_model_predictions_' + model + '_tau_start_' + str(
                    tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            pd_difference_per_row = pd.read_pickle(
                'model_predictions/' + 'pd_difference_in_model_predictions_' + model + '_tau_start_' + str(
                    tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

        pd_most_contrastive_trials = get_most_contrastive_trials(pd_difference_per_row, numbering, model, WordNo, speaker_rationality=speaker_rationality, listener_rationality=listener_rationality)
        pd.set_option('display.max_columns', None)
        print("pd_most_contrastive_trials is:")
        print(pd_most_contrastive_trials)



print('')
print('')
print('Most contrastive trials ACROSS PARAMETERS:')
contrastive_trials_across_parameters(baseline_models, words)



print('')
print('')
print('')
print('')
print('Most contrastive trials USING BEST-FITTING PARAMETERS:')
best_fit_parameters_exp2_dict = {"English":[1.7, 1.15],
                                 "Italian":[0.65, 1.],
                                 "Portuguese":[0.55, 1.65],
                                 "Spanish":[0.5, 1.95]}

contrastive_trials_for_best_parameters(best_fit_parameters_exp2_dict, languages)
