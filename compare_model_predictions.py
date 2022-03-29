import numpy as np
import pandas as pd


# PARAMETER SETTINGS: #
languages = ["English", "Italian", "Portuguese", "Spanish"]
object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions
listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
listener_attentions = [0, 1, 2, 3]  # array of all possible listener attentions
tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

measure = "probability"  # can be set to either "difference" or "probability"

all_combos = True  # Can be set to either True (to load in predictions for all object_position*listener_attention combinations) or False

# FUNCTION DEFINITIONS: #


def get_model_predictions(pd_model_predictions_baseline, pd_model_predictions_attention, model, language, speaker_tau, listener_tau, object_pos, listener_pos, listener_att):

    if language == "English" or language == "Italian":
        WordNo = 2
        words = ["este", "aquel"]
    elif language == "Portuguese" or language == "Spanish":
        WordNo = 3
        words = ["este", "ese", "aquel"]
    probs_per_word_baseline = np.zeros((WordNo))
    probs_per_word_attention = np.zeros((WordNo))
    for i in range(len(words)):
        word = words[i]

        model_prediction_row_baseline = pd_model_predictions_baseline[pd_model_predictions_baseline["Model"]==model][pd_model_predictions_baseline["Word"]==word][pd_model_predictions_baseline["Referent"]==object_pos][pd_model_predictions_baseline["Listener_pos"]==listener_pos][pd_model_predictions_baseline["Listener_att"]==listener_att][pd_model_predictions_baseline["WordNo"]==WordNo][pd_model_predictions_baseline["SpeakerTau"]==speaker_tau][pd_model_predictions_baseline["ListenerTau"]==listener_tau]
        model_prediction_prob_baseline = model_prediction_row_baseline["Probability"]
        probs_per_word_baseline[i] = model_prediction_prob_baseline

        model_prediction_row_attention = pd_model_predictions_attention[pd_model_predictions_attention["Model"]==model+"_attention"][pd_model_predictions_attention["Word"]==word][pd_model_predictions_attention["Referent"]==object_pos][pd_model_predictions_baseline["Listener_pos"]==listener_pos][pd_model_predictions_attention["Listener_att"]==listener_att][pd_model_predictions_attention["WordNo"]==WordNo][pd_model_predictions_attention["SpeakerTau"]==speaker_tau][pd_model_predictions_attention["ListenerTau"]==listener_tau]
        model_prediction_prob_attention = model_prediction_row_attention["Probability"]
        probs_per_word_attention[i] = model_prediction_prob_attention

    return probs_per_word_baseline, probs_per_word_attention



def subtract_model_predictions(pd_model_predictions_baseline, pd_model_predictions_attention, model, language, speaker_tau, listener_tau, object_pos, listener_att):

    if language == "English" or language == "Italian":
        WordNo = 2
        words = ["este", "aquel"]
    elif language == "Portuguese" or language == "Spanish":
        WordNo = 3
        words = ["este", "ese", "aquel"]
    probs_per_word_baseline = np.zeros((WordNo))
    probs_per_word_attention = np.zeros((WordNo))
    for i in range(len(words)):
        word = words[i]

        model_prediction_row_baseline = pd_model_predictions_baseline[pd_model_predictions_baseline["Model"]==model][pd_model_predictions_baseline["Word"]==word][pd_model_predictions_baseline["Referent"]==object_pos][pd_model_predictions_baseline["Listener_att"]==listener_att][pd_model_predictions_baseline["WordNo"]==WordNo][pd_model_predictions_baseline["SpeakerTau"]==speaker_tau][pd_model_predictions_baseline["ListenerTau"]==listener_tau]
        model_prediction_prob_baseline = model_prediction_row_baseline["Probability"]
        probs_per_word_baseline[i] = model_prediction_prob_baseline

        model_prediction_row_attention = pd_model_predictions_attention[pd_model_predictions_attention["Model"]==model+"_attention"][pd_model_predictions_attention["Word"]==word][pd_model_predictions_attention["Referent"]==object_pos][pd_model_predictions_attention["Listener_att"]==listener_att][pd_model_predictions_attention["WordNo"]==WordNo][pd_model_predictions_attention["SpeakerTau"]==speaker_tau][pd_model_predictions_attention["ListenerTau"]==listener_tau]
        model_prediction_prob_attention = model_prediction_row_attention["Probability"]
        probs_per_word_attention[i] = model_prediction_prob_attention

    difference = np.subtract(probs_per_word_attention, probs_per_word_baseline)

    return np.round(difference, decimals=4)



def predictions_over_situations(measure, pd_model_predictions_baseline, pd_model_predictions_attention, model, language, WordNo, speaker_tau, listener_tau, object_positions, listener_positions, listener_attentions):
    if measure == "probability":
        if WordNo == 2:
            output_dict = {"model_variant":[],
                            "object_pos": [],
                           "listener_pos": [],
                           "listener_att": [],
                           "situation": [],
                           measure + "_este": [],
                           measure + "_aquel": []}
        elif WordNo == 3:
            output_dict = {"model_variant":[],
                            "object_pos": [],
                           "listener_pos": [],
                           "listener_att": [],
                           "situation": [],
                           measure + "_este": [],
                           measure + "_ese": [],
                           measure + "_aquel": []}

    elif measure == "difference":
        if WordNo == 2:
            output_dict = {"object_pos":[],
                           "listener_pos": [],
                           "listener_att":[],
                            "situation":[],
                           measure+"_este":[],
                            measure + "_aquel":[]}
        elif WordNo == 3:
            output_dict = {"object_pos":[],
                           "listener_pos": [],
                           "listener_att":[],
                            "situation": [],
                           measure+"_este":[],
                            measure+"_ese":[],
                            measure+"_aquel":[]}

    for object_pos in object_positions:
        for listener_pos in listener_positions:
            for listener_att in listener_attentions:
                output_dict["object_pos"].append(object_pos)
                output_dict["listener_pos"].append(listener_pos)
                output_dict["listener_att"].append(listener_att)
                if measure == "probability":
                    output_dict["object_pos"].append(object_pos)
                    output_dict["listener_pos"].append(listener_pos)
                    output_dict["listener_att"].append(listener_att)
                if listener_att - object_pos == 2:
                    situation = "2 too far"
                elif listener_att - object_pos == 1:
                    situation = "1 too far"
                elif listener_att - object_pos == 0:
                    situation = "aligned"
                elif listener_att - object_pos == -1:
                    situation = "1 too close"
                elif listener_att - object_pos == -2:
                    situation = "2 too close"
                elif listener_att - object_pos == -3:
                    situation = "3 too close"
                output_dict["situation"].append(situation)
                if measure == "probability":
                    output_dict["situation"].append(situation)

                if measure == "probability":
                    probs_per_word_baseline, probs_per_word_attention = get_model_predictions(pd_model_predictions_baseline, pd_model_predictions_attention, model, language, speaker_tau, listener_tau, object_pos, listener_pos, listener_att)
                    output_dict["model_variant"].append("Baseline")
                    if WordNo == 2:
                        output_dict[measure+"_este"].append(probs_per_word_baseline[0])
                        output_dict[measure+"_aquel"].append(probs_per_word_baseline[1])
                    elif WordNo == 3:
                        output_dict[measure+"_este"].append(probs_per_word_baseline[0])
                        output_dict[measure+"_ese"].append(probs_per_word_baseline[1])
                        output_dict[measure+"_aquel"].append(probs_per_word_baseline[2])
                    output_dict["model_variant"].append("Attention-correction")
                    if WordNo == 2:
                        output_dict[measure+"_este"].append(probs_per_word_attention[0])
                        output_dict[measure+"_aquel"].append(probs_per_word_attention[1])
                    elif WordNo == 3:
                        output_dict[measure+"_este"].append(probs_per_word_attention[0])
                        output_dict[measure+"_ese"].append(probs_per_word_attention[1])
                        output_dict[measure+"_aquel"].append(probs_per_word_attention[2])

                elif measure == "difference":
                    output = subtract_model_predictions(pd_model_predictions_baseline, pd_model_predictions_attention, model, language, speaker_tau, listener_tau, object_pos, listener_att)
                    if WordNo == 2:
                        output_dict[measure+"_este"].append(output[0])
                        output_dict[measure+"_aquel"].append(output[1])
                    elif WordNo == 3:
                        output_dict[measure+"_este"].append(output[0])
                        output_dict[measure+"_ese"].append(output[1])
                        output_dict[measure+"_aquel"].append(output[2])

    pd_output_per_situation = pd.DataFrame.from_dict(output_dict)
    return pd_output_per_situation



def prediction_difference_across_parameter_settings(measure, model, language, WordNo, object_positions, listener_positions, listener_attentions, tau_start, tau_stop, tau_step):
    if WordNo == 2:
        difference_across_parameters_dict = {"SpeakerTau":[],
                           "ListenerTau":[],
                        "object_pos":[],
                        "listener_pos": [],
                       "listener_att":[],
                        "situation":[],
                       measure+"_este":[],
                        measure+"_aquel":[]}
    elif WordNo == 3:
        difference_across_parameters_dict = {"SpeakerTau":[],
                           "ListenerTau":[],
                        "object_pos":[],
                        "listener_pos": [],
                       "listener_att":[],
                        "situation": [],
                       measure+"_este":[],
                        measure+"_ese":[],
                        measure+"_aquel":[]}
    pd_difference_across_parameters = pd.DataFrame.from_dict(difference_across_parameters_dict)
    for listener_rationality in np.arange(tau_start, tau_stop, tau_step):
        print('')
        print(f"listener_rationality is {listener_rationality}:")
        for speaker_rationality in np.arange(tau_start, tau_stop, tau_step):
            print(f"speaker_rationality is {speaker_rationality}:")

            if all_combos is True:
                n_situations = len(object_positions) * len(listener_positions) * len(listener_attentions)
            else:
                n_situations = len(object_positions)*len(listener_attentions)

            listener_tau_list = []
            speaker_tau_list = []
            if measure == "difference":
                for _ in range(n_situations):
                    listener_tau_list.append(round(listener_rationality, 2))
                    speaker_tau_list.append(round(speaker_rationality, 2))
            elif measure == "probability":
                for _ in range(n_situations*2):
                    listener_tau_list.append(round(listener_rationality, 2))
                    speaker_tau_list.append(round(speaker_rationality, 2))


            # LOAD IN MODEL PREDICTIONS:
            models_for_filename = ["distance", "person"]
            if all_combos is True:
                model_predictions_baseline = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_All_Combos_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
            else:
                model_predictions_baseline = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

            models_for_filename = ["distance_attention", "person_attention"]
            if all_combos is True:
                model_predictions_attention = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_All_Combos_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
            else:
                model_predictions_attention = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

            pd_difference_per_situation = predictions_over_situations(measure, model_predictions_baseline, model_predictions_attention, model, language, WordNo, round(speaker_rationality, 2), round(listener_rationality, 2), object_positions, listener_positions, listener_attentions)

            pd_difference_per_situation.insert(0, "ListenerTau", listener_tau_list)
            pd_difference_per_situation.insert(0, "SpeakerTau", speaker_tau_list)

            pd_difference_across_parameters = pd.concat([pd_difference_across_parameters, pd_difference_per_situation])

    return pd_difference_across_parameters


best_fit_parameters_exp1_dict = {"English":[0.65, 1.15],
                                 "Italian":[0.5, 0.5],
                                 "Portuguese":[0.45, 0.95],
                                 "Spanish":[0.5, 0.8]}

best_fit_parameters_exp2_dict = {"English":[1.7, 1.15],
                                 "Italian":[0.65, 1.],
                                 "Portuguese":[0.55, 1.65],
                                 "Spanish":[0.5, 1.95]}


for exp in [1, 2]:
    if exp == 1:
        parameter_dict = best_fit_parameters_exp1_dict
    elif exp == 2:
        parameter_dict = best_fit_parameters_exp2_dict
    print('')
    print('')
    print('')
    print("parameter_dict is:")
    print(parameter_dict)

    for language in languages:
        if language == "English" or language == "Italian":
            model = "distance"
            WordNo = 2
        elif language == "Portuguese" or language == "Spanish":
            model = "person"
            WordNo = 3
        print('')
        print('')
        print('')
        print(f"LANGUAGE = {language} + MODEL = {model}:")

        speaker_tau = parameter_dict[language][0]
        listener_tau = parameter_dict[language][1]
        print('')
        print("speaker_tau is:")
        print(speaker_tau)
        print("listener_tau is:")
        print(listener_tau)

        # LOAD IN MODEL PREDICTIONS:

        models_for_filename = ["distance", "person"]
        if all_combos is True:
            model_predictions_baseline = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_All_Combos_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
        else:
            model_predictions_baseline = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

        models_for_filename = ["distance_attention", "person_attention"]
        if all_combos is True:
            model_predictions_attention = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_All_Combos_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
        else:
            model_predictions_attention = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

        pd_difference_per_situation = predictions_over_situations(measure, model_predictions_baseline, model_predictions_attention, model, language, WordNo, speaker_tau, listener_tau, object_positions, listener_positions, listener_attentions)

        pd.set_option('display.max_columns', None)
        print('')
        print("pd_difference_per_situation is:")
        print(pd_difference_per_situation)

        if all_combos is True:
            pd_difference_per_situation.to_pickle(
                'model_predictions/' + 'pd_' + measure + '_All_Combos_' + 'exp_' + str(exp) + '_' + language + '_' + model + '_speaker_tau_'+ str(speaker_tau) + '_listener_tau_' + str(listener_tau) + '.pkl')
        else:
            pd_difference_per_situation.to_pickle(
                'model_predictions/' + 'pd_' + measure + 'exp_' + str(exp) + '_' + language + '_' + model + '_speaker_tau_'+ str(speaker_tau) + '_listener_tau_' + str(listener_tau) + '.pkl')




for language in languages:
    if language == "English" or language == "Italian":
        model = "distance"
        WordNo = 2
    elif language == "Portuguese" or language == "Spanish":
        model = "person"
        WordNo = 3
    print('')
    print('')
    print('')
    print(f"LANGUAGE = {language} + MODEL = {model}:")

    pd_difference_across_parameters = prediction_difference_across_parameter_settings(measure, model, language, WordNo, object_positions, listener_positions, listener_attentions, tau_start, tau_stop, tau_step)
    pd.set_option('display.max_columns', None)
    print('')
    print('')
    print("pd_difference_across_parameters is:")
    print(pd_difference_across_parameters)

    if all_combos is True:
        pd_difference_across_parameters.to_pickle('model_predictions/' + 'pd_'+measure+'_across_parameters_All_Combos_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    else:
        pd_difference_across_parameters.to_pickle('model_predictions/' + 'pd_'+measure+'_across_parameters_Attention_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')