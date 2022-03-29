import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
baseline_models = ["distance", "person"]
words = [2, 3]
object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions
listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
listener_attentions = [0, 1, 2, 3]  # array of all possible listener attentions
tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

absolute = False

all_combos = True  # Can be set to either True (to load in predictions for all object_position*listener_attention combinations) or False


# FUNCTION DEFINITIONS: #


def get_model_predictions(pd_model_predictions_baseline, pd_model_predictions_attention, model, language, speaker_tau, listener_tau, object_pos, listener_att):

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

    return probs_per_word_baseline, probs_per_word_attention



def calc_most_different_situation(probs_per_word_baseline, probs_per_word_attention, absolute=False):
    difference_per_row_dict = {"Model":[],
                               "WordNo":[],
                               "SpeakerTau":[],
                               "ListenerTau":[],
                               "Speaker_pos":[],
                               "Listener_pos":[],
                               "Listener_att":[],
                               "Referent":[],
                               "Word":[],
                               "Prob_difference":[]}
    for i in range(0, len(probs_per_word_baseline.index)):
        row_baseline = probs_per_word_baseline.iloc[[i]]
        row_attention = probs_per_word_attention.iloc[[i]]
        if row_baseline["Model"][i] not in row_attention["Model"][i]:
            raise ValueError('Model type for Baseline dataframe does not match model type for Attention-correction dataframe')
        if row_baseline["WordNo"][i] != row_attention["WordNo"][i]:
            raise ValueError('WordNo does not match between Baseline and Attention-correction dataframe')
        if row_baseline["SpeakerTau"][i] != row_attention["SpeakerTau"][i]:
            raise ValueError('SpeakerTau does not match between Baseline and Attention-correction dataframe')
        if row_baseline["ListenerTau"][i] != row_attention["ListenerTau"][i]:
            raise ValueError('ListenerTau does not match between Baseline and Attention-correction dataframe')
        if row_baseline["Speaker_pos"][i] != row_attention["Speaker_pos"][i]:
            raise ValueError('Speaker_pos does not match between Baseline and Attention-correction dataframe')
        if row_baseline["Listener_pos"][i] != row_attention["Listener_pos"][i]:
            raise ValueError('Listener_pos does not match between Baseline and Attention-correction dataframe')
        if row_baseline["Listener_att"][i] != row_attention["Listener_att"][i]:
            raise ValueError('Listener_att does not match between Baseline and Attention-correction dataframe')
        if row_baseline["Referent"][i] != row_attention["Referent"][i]:
            raise ValueError('Referent (i.e. object position) does not match between Baseline and Attention-correction dataframe')
        if row_baseline["Word"][i] != row_attention["Word"][i]:
            raise ValueError('Word does not match between Baseline and Attention-correction dataframe')

        difference_per_row_dict["Model"].append(row_baseline["Model"][i])
        difference_per_row_dict["WordNo"].append(row_baseline["WordNo"][i])
        difference_per_row_dict["SpeakerTau"].append(row_baseline["SpeakerTau"][i])
        difference_per_row_dict["ListenerTau"].append(row_baseline["ListenerTau"][i])
        difference_per_row_dict["Speaker_pos"].append(row_baseline["Speaker_pos"][i])
        difference_per_row_dict["Listener_pos"].append(row_baseline["Listener_pos"][i])
        difference_per_row_dict["Listener_att"].append(row_baseline["Listener_att"][i])
        difference_per_row_dict["Referent"].append(row_baseline["Referent"][i])
        difference_per_row_dict["Word"].append(row_baseline["Word"][i])

        prob_difference_attention_minus_baseline = row_attention["Probability"][i] - row_baseline["Probability"][i]
        if absolute is True:
            difference_per_row_dict["Prob_difference"].append(round(np.abs(prob_difference_attention_minus_baseline), 4))
        else:
            difference_per_row_dict["Prob_difference"].append(round(prob_difference_attention_minus_baseline, 4))

    pd_difference_per_row = pd.DataFrame.from_dict(difference_per_row_dict)

    return pd_difference_per_row




def plot_scatter_correlate_model_predictions(data_selection_pd, model, words):
    sns.set_theme(style="whitegrid")
    sns.color_palette("colorblind")

    if words == 2:
        my_colors = sns.color_palette("colorblind", 2)
    elif words == 3:
        my_colors = sns.color_palette("colorblind", 3)
        my_order = [0, 2, 1]
        my_colors =  [my_colors[i] for i in my_order]

    sns.scatterplot(data=data_selection_pd, x="Probability_Baseline", y="Probability_Attention", hue="Word", palette=my_colors)

    plt.title(f"Predictions Baseline against Attention-correction: {model} + {words}")
    if all_combos is True:
        plt.savefig('plots/'+'scatterplot_correlate_model_predictions_All_Combos_'+model+'_'+str(words)+'_words'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    else:
        plt.savefig('plots/'+'scatterplot_correlate_model_predictions_Attention_'+model+'_'+str(words)+'_words'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()


def plot_relplot_correlate_model_predictions(data_selection_pd, model, words):
    sns.set_theme(style="whitegrid")
    sns.color_palette("colorblind")

    situation_order = ["3 too close", "2 too close", "1 too close", "aligned", "1 too far", "2 too far"]

    if words == 2:
        my_colors = sns.color_palette("colorblind", 2)
    elif words == 3:
        my_colors = sns.color_palette("colorblind", 3)
        my_order = [0, 2, 1]
        my_colors =  [my_colors[i] for i in my_order]

    sns.relplot(x="Probability_Baseline", y="Probability_Attention", hue="Word", col="Situation", col_order=situation_order, palette=my_colors, data=data_selection_pd)

    # plt.suptitle(f"Predictions Baseline against Attention-correction: {model} + {words}")
    plt.suptitle(f"{model} + {words}")

    # plt.tight_layout()
    if all_combos is True:
        plt.savefig('plots/'+'relplot_correlate_model_predictions_All_Combos_'+model+'_'+str(words)+'_words'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    else:
        plt.savefig('plots/'+'relplot_correlate_model_predictions_Attention_'+model+'_'+str(words)+'_words'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()




for model in baseline_models:
    print('')
    print('')
    print('')
    print(f"MODEL = {model}:")

    # LOAD IN MODEL PREDICTIONS:

    if all_combos is True:
        models_for_filename = ["distance", "person"]
        model_predictions_baseline = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_All_Combos_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

        models_for_filename = ["distance_attention", "person_attention"]
        model_predictions_attention = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_All_Combos_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
    else:
        models_for_filename = ["distance", "person"]
        model_predictions_baseline = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

        models_for_filename = ["distance_attention", "person_attention"]
        model_predictions_attention = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

    pd.set_option('display.max_columns', None)


    print('')
    print('')
    print("model_predictions_baseline are:")
    print(model_predictions_baseline)


    print('')
    print('')
    print("model_predictions_attention are:")
    print(model_predictions_attention)



    pd_difference_per_row = calc_most_different_situation(model_predictions_baseline, model_predictions_attention, absolute=absolute)
    print('')
    print('')
    print("pd_difference_per_row is:")
    print(pd_difference_per_row)

    if all_combos is True:
        if absolute is True:
            pd_difference_per_row.to_pickle('model_predictions/' + 'pd_abs_difference_in_model_predictions_All_Combos_'+ model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            pd_difference_per_row.to_pickle('model_predictions/' + 'pd_difference_in_model_predictions_All_Combos_'+ model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    else:
        if absolute is True:
            pd_difference_per_row.to_pickle('model_predictions/' + 'pd_abs_difference_in_model_predictions_'+ model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        else:
            pd_difference_per_row.to_pickle('model_predictions/' + 'pd_difference_in_model_predictions_'+ model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')



    attention_probability_column = model_predictions_attention["Probability"]
    print('')
    print('')
    print("attention_probability_column is:")
    print(attention_probability_column)
    print("len(attention_probability_column) is:")
    print(len(attention_probability_column))

    predictions_both_models = model_predictions_baseline
    predictions_both_models["Probability_Attention"] = attention_probability_column
    print('')
    print('')
    print("predictions_both_models is:")
    print(predictions_both_models)

    predictions_both_models.rename(columns={'Probability': 'Probability_Baseline'}, inplace=True)
    print('')
    print('')
    print("predictions_both_models is:")
    print(predictions_both_models)


    data_selection = predictions_both_models[predictions_both_models["Model"] == model]
    print('')
    print('')
    print("data_selection is:")
    print(data_selection)


    situation_column = []
    for index, row in data_selection.iterrows():
        listener_att = row['Listener_att']
        object_pos = row['Referent']
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
        situation_column.append(situation)

    print('')
    print('')
    print("situation_column is:")
    print(situation_column)

    data_selection["Situation"] = situation_column
    print('')
    print('')
    print("data_selection is:")
    print(data_selection)


    for WordNo in words:

        subset_pd = data_selection[data_selection["WordNo"]==WordNo]
        print('')
        print('')
        print("subset_pd is:")
        print(subset_pd)

        plot_scatter_correlate_model_predictions(subset_pd, model, WordNo)

        plot_relplot_correlate_model_predictions(subset_pd, model, WordNo)