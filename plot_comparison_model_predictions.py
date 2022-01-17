import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
# languages = ["English", "Italian", "Portuguese", "Spanish"]
languages = ["English", "Portuguese"]
object_positions = [1, 2, 3]  # array of all possible object (= referent) positions
listener_attentions = [0, 1, 2, 3]  # array of all possible listener positions
tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

measure = "probability"  # can be set to either "difference" or "probability"

# FUNCTION DEFINITIONS: #


def plot_displot_probabilities(measure, pd_across_parameters, relevant_word, language, WordNo, model, tau_start, tau_stop, tau_step, ymin=None, ymax=None):


    sns.set_theme(style="whitegrid")
    sns.color_palette("colorblind")

    # if relevant_word == "este":
    #     data_selection = pd_difference_across_parameters[pd_difference_across_parameters["situation"] == "2 too far"]
    #     print('')
    #     print('')
    #     print("data_selection is:")
    #     print(data_selection)
    # elif relevant_word == "aquel":
    #     data_selection = pd_difference_across_parameters[pd_difference_across_parameters["situation"] == "2 too close"]
    #     print('')
    #     print('')
    #     print("data_selection is:")
    #     print(data_selection)


    # data_selection = pd_difference_across_parameters[(pd_difference_across_parameters["SpeakerTau"] > 0.4) & (pd_difference_across_parameters["SpeakerTau"] < 1.75)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN SPEAKER TAU is:")
    # print(data_selection)
    #
    # data_selection = data_selection[(data_selection["ListenerTau"] > 0.85) & (data_selection["ListenerTau"] < 2.0)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN LISTENER TAU is:")
    # print(data_selection)


    # data_selection = pd_across_parameters[(pd_across_parameters["SpeakerTau"] > 0.45) & (pd_across_parameters["SpeakerTau"] < 0.6)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN SPEAKER TAU is:")
    # print(data_selection)
    #
    # data_selection = data_selection[(data_selection["ListenerTau"] > 0.95) & (data_selection["ListenerTau"] < 1.1)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN LISTENER TAU is:")
    # print(data_selection)

    print('')
    print("len(pd_across_parameters) is:")
    print(len(pd_across_parameters))
    index_list = range(0, len(pd_across_parameters))
    print('')
    print("index_list is:")
    print(index_list)

    pd_across_parameters.index = index_list
    print('')
    print('')
    print("pd_across_parameters AFTER SETTING INDEX is:")
    print(pd_across_parameters)

    sns.displot(data=pd_across_parameters, x="probability_" + relevant_word, col="situation", row="model_variant")

    # data_selection = data_selection[data_selection["model_variant"] == "Baseline"]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN TO ONE MODEL VARIANT is:")
    # print(data_selection)

    # sns.histplot(data=data_selection, x="probability_" + relevant_word, hue="situation")

    # if ymin != None and ymax != None:
    #     ax.set(ylim=(ymin, ymax))
    # ax.set(ylabel=f"Avg. {relevant_word} prob. difference: Attention - Baseline")

    # plt.title(f"Increase in {relevant_word} prob. Attention vs. Baseline: {model} + {WordNo}-word")
    plt.savefig('plots/'+'displot_model_predictions_'+measure+'_Attention_'+language+'_'+model+'_'+relevant_word+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()



def plot_barplot_differences(measure, pd_difference_across_parameters, relevant_word, language, WordNo, model, tau_start, tau_stop, tau_step, ymin=None, ymax=None):


    sns.set_theme(style="whitegrid")
    sns.color_palette("colorblind")

    # if relevant_word == "este":
    #     data_selection = pd_difference_across_parameters[pd_difference_across_parameters["situation"] == "2 too far"]
    #     print('')
    #     print('')
    #     print("data_selection is:")
    #     print(data_selection)
    # elif relevant_word == "aquel":
    #     data_selection = pd_difference_across_parameters[pd_difference_across_parameters["situation"] == "2 too close"]
    #     print('')
    #     print('')
    #     print("data_selection is:")
    #     print(data_selection)


    # data_selection = data_selection[(data_selection["SpeakerTau"] > 0.4) & (data_selection["SpeakerTau"] < 1.75)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN SPEAKER TAU is:")
    # print(data_selection)
    #
    # data_selection = data_selection[(data_selection["ListenerTau"] > 0.85) & (data_selection["ListenerTau"] < 2.0)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN LISTENER TAU is:")
    # print(data_selection)


    # data_selection = data_selection[(data_selection["SpeakerTau"] > 0.45) & (data_selection["SpeakerTau"] < 1.75)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN SPEAKER TAU is:")
    # print(data_selection)
    #
    # data_selection = data_selection[(data_selection["ListenerTau"] > 0.95) & (data_selection["ListenerTau"] < 2.0)]
    # print('')
    # print('')
    # print("data_selection AFTER NARROWING DOWN LISTENER TAU is:")
    # print(data_selection)

    # ax = sns.histplot(data=data_selection, x="difference_"+relevant_word, hue="situation")

    # sns.histplot(data=data_selection, x="difference_" + relevant_word)

    ax = sns.barplot(x="situation", y="difference_" + relevant_word, data=pd_difference_across_parameters)
    if ymin != None and ymax != None:
        ax.set(ylim=(ymin, ymax))
    ax.set(ylabel=f"Avg. {relevant_word} prob. difference: Attention - Baseline")

    plt.title(f"Increase in {relevant_word} prob. Attention vs. Baseline: {model} + {WordNo}-word")
    plt.savefig('plots/'+'barplot_model_predictions_'+measure+'_Attention_'+language+'_'+model+'_'+relevant_word+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.pdf')
    plt.show()




ymin_este = 0
ymax_este = 0
ymin_aquel = 0
ymax_aquel = 0
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


    pd_across_parameters = pd.read_pickle('model_predictions/' + 'pd_' + measure + '_across_parameters_Attention_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

    pd.set_option('display.max_columns', None)
    print('')
    print('')
    print("pd_across_parameters is:")
    print(pd_across_parameters)


    data_selection_este = pd_across_parameters[(pd_across_parameters["situation"] == "1 too far") | (pd_across_parameters["situation"] == "2 too far")]
    print('')
    print('')
    print("data_selection_este is:")
    print(data_selection_este)

    min_este = np.amin(data_selection_este[measure+"_este"])
    print('')
    print('')
    print("min_este is:")
    print(min_este)
    if min_este < ymin_este:
        ymin_este = min_este
    print("ymin_este is:")
    print(ymin_este)

    max_este = np.amax(data_selection_este[measure+"_este"])
    print('')
    print('')
    print("max_este is:")
    print(max_este)
    if max_este > ymax_este:
        ymax_este = max_este
    print("ymax_este is:")
    print(ymax_este)


    data_selection_aquel = pd_across_parameters[(pd_across_parameters["situation"] == "1 too close") | (pd_across_parameters["situation"] == "2 too close") | (pd_across_parameters["situation"] == "3 too close")]
    print('')
    print('')
    print("data_selection_aquel is:")
    print(data_selection_aquel)

    min_aquel = np.amin(data_selection_aquel[measure+"_aquel"])
    print('')
    print('')
    print("min_aquel is:")
    print(min_aquel)
    if min_aquel < ymin_aquel:
        ymin_aquel = min_aquel
    print("ymin_aquel is:")
    print(ymin_aquel)

    max_este = np.amax(data_selection_aquel[measure+"_aquel"])
    print('')
    print('')
    print("max_este is:")
    print(max_este)
    if max_este > ymax_este:
        ymax_este = max_este
    print("ymax_este is:")
    print(ymax_este)



    print('')
    print('')
    print("ymin_este is:")
    print(ymin_este)
    print("ymax_este is:")
    print(ymax_este)
    print('')
    print("ymin_aquel is:")
    print(ymin_aquel)
    print("ymax_este is:")
    print(ymax_este)

    if measure == "probability":

        plot_displot_probabilities(measure, data_selection_este, "este", language, WordNo, model, tau_start, tau_stop, tau_step)

        plot_displot_probabilities(measure, data_selection_aquel, "aquel", language, WordNo, model, tau_start, tau_stop, tau_step)


    elif measure == "difference":

        ymin = -0.025  # TODO: Should this really be hardcoded?
        ymax = 0.12

        plot_barplot_differences(measure, data_selection_este, "este", language, WordNo, model, tau_start, tau_stop, tau_step, ymin=ymin, ymax=ymax)

        plot_barplot_differences(measure, data_selection_aquel, "aquel", language, WordNo, model, tau_start, tau_stop, tau_step, ymin=ymin, ymax=ymax)