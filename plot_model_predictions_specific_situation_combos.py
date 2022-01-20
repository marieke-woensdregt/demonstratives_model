import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# PARAMETER SETTINGS: #
experiment = "baseline"

if experiment == "attention":
    models = ["distance_attention", "person_attention"]
else:
    models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]
if experiment == "attention":
    object_positions = [1, 2, 3]  # array of all possible object (= referent) positions
else:
    object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions

listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
listener_attentions = [0, 1, 2, 3]  # array of all possible listener positions

tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

if experiment == "attention":
    object_positions_of_interest = [0, 0, 3, 3]
    listener_attentions_of_interest = [0, 0, 1, 1]
else:
    object_positions_of_interest = [0, 0, 3, 3]
    listener_positions_of_interest = [0, 0, 1, 1]
transparent_plots = False  # Can be set to True or False






def get_probs_and_proportions_for_situation(experiment, pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau, object_pos, listener_pos=None, listener_att=None):

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
        else:
            data_count_row = pd_data[pd_data["Object_Position"] == object_pos+1][pd_data["Listener_Position"] == listener_pos+1][pd_data["Language"] == language]
        if experiment == "attention":
            data_count = data_count_row[word.title()]
        else:
            data_count = data_count_row[word]

        if len(data_count) > 1:
            data_count = np.mean(data_count)  #TODO: Check whether this is really the right way to handle this. For some reason these rows are repeated in the data files (twice for the 2-word languages, and thrice for the 3-word languages), due to my bad R skills for doing data preprocessing...
            counts_per_word[i] = data_count
        else:
            counts_per_word[i] = data_count

        total = data_count_row["Total"]
        if len(total) > 1:
            total = np.mean(total)

    print('')
    print('')
    print("probs_per_word are:")
    print(probs_per_word)
    print("np.sum(probs_per_word) are:")
    print(np.sum(probs_per_word))
    print('')
    print("counts_per_word is:")
    print(counts_per_word)
    print('')
    print("total is:")
    print(total)
    print("int(total) is:")
    print(int(total))

    proportions_per_word = np.divide(np.array(counts_per_word), int(total))
    print('')
    print("proportions_per_word is:")
    print(proportions_per_word)
    print("np.sum(proportions_per_word) is:")
    print(np.sum(proportions_per_word))
    return probs_per_word, proportions_per_word




def plot_stacked_bar_across_model_and_data(experiment, pd_word_probs_over_situations, language, model, origin, object_positions_of_interest, listener_positions_of_interest=None, listener_attentions_of_interest=None):
    # set seaborn plotting aesthetics
    if transparent_plots is True:
        sns.set(style='white')
    else:
        sns.set(style='whitegrid')

    pd.set_option('display.max_columns', None)

    pd_selection = pd_word_probs_over_situations[pd_word_probs_over_situations["origin"] == origin]
    print('')
    print('')
    print("pd_selection is:")
    print(pd_selection)

    if language == "English" or language == "Italian":
        my_colors = sns.color_palette("colorblind", 2)
    elif language == "Portuguese" or language == "Spanish":
        my_colors = sns.color_palette("colorblind", 3)
        my_order = [0, 2, 1]
        my_colors =  [my_colors[i] for i in my_order]

    if experiment == "attention":
        ax = pd_selection.set_index('listener_att').plot(kind='bar', stacked=True, color=my_colors)
    else:
        ax = pd_selection.set_index('listener_pos').plot(kind='bar', stacked=True, color=my_colors)

    # add overall title
    origin_for_title = origin
    origin_for_title = origin_for_title.title()
    model_for_title = model
    model_for_title = model_for_title.title()
    model_for_title = model_for_title.replace("_", "+")

    if transparent_plots is True:
        plt.title("")
    else:
        if origin == "model predictions":
            plt.title(f"{origin_for_title} for {model_for_title}: {language}", fontsize=17, x=0.6)
        elif origin == "human participants":
            if experiment == "baseline":
                plt.title(f"{origin_for_title} for Experiment 1: {language}", fontsize=17, x=0.6)
            elif experiment == "attention":
                plt.title(f"{origin_for_title} for Experiment 2: {language}", fontsize=17, x=0.6)

    if transparent_plots is True:
        plt.xlabel("")
        plt.ylabel("")
    else:
        # add axis titles
        if experiment == "attention":
            plt.xlabel('Object position + Listener attention', fontsize=16)
        else:
            plt.xlabel('Object position + Listener position', fontsize=16)
        if origin == "model predictions":
            plt.ylabel('Probability', fontsize=16)
        elif origin == "human participants":
            plt.ylabel('Proportion', fontsize=16)

    new_x_tick_labels = []
    if experiment == "attention":
        for i in range(len(object_positions_of_interest)):
            object_pos = object_positions_of_interest[i]
            listener_att = listener_attentions_of_interest[i]
            xtick_label = str(object_pos+1)+' + '+str(listener_att+1) # +1 in order to adhere to the experimental numbering
            new_x_tick_labels.append(xtick_label)
    else:
        for i in range(len(object_positions_of_interest)):
            object_pos = object_positions_of_interest[i]
            listener_pos = listener_positions_of_interest[i]
            xtick_label = str(object_pos+1)+' + '+str(listener_pos+1) # +1 in order to adhere to the experimental numbering
            new_x_tick_labels.append(xtick_label)

    plt.xticks([0, 1, 2, 3], new_x_tick_labels, rotation=0, fontsize=15)
    plt.yticks(fontsize=15)

    if transparent_plots is True:
        plt.tick_params(axis="x", colors="white")
        plt.tick_params(axis="y", colors="white")
        plt.ylim(0.0, 1.0)

    if transparent_plots is True:
        plt.legend().remove()
    else:
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, prop={"size":16})
    plt.tight_layout()
    if experiment == "attention":
        filename = 'plots/'+'stacked_bars_'+experiment+'_object_pos_'+str(object_positions_of_interest).replace(" ", "")+'_listener_att_'+str(listener_attentions_of_interest).replace(" ", "")+'_'+origin.replace(" ", "")+'_'+language+'_'+model
    else:
        filename = 'plots/'+'stacked_bars_'+experiment+'_object_pos_'+str(object_positions_of_interest).replace(" ", "")+'_listener_pos_'+str(listener_positions_of_interest).replace(" ", "")+'_'+origin.replace(" ", "")+'_'+language+'_'+model
    if transparent_plots is True:
        filename = filename+'.png'
    else:
        filename = filename+'.pdf'
    plt.savefig(filename, transparent=transparent_plots)
    plt.show()



if __name__ == "__main__":
    best_fit_parameters_exp1_dict = {"English":[0.75, 0.95],
                                     "Italian":[0.45, 1.2],
                                     "Portuguese":[0.45, 0.9],
                                     "Spanish":[0.45, 0.9]}

    best_fit_parameters_exp2_dict = {"English":[1.7, 1.15],
                                     "Italian":[0.65, 1.],
                                     "Portuguese":[0.55, 1.65],
                                     "Spanish":[0.5, 1.95]}

    for language in languages:
        # print('')
        # print('')
        # print(language)

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

        if experiment == "attention":
            best_fit_parameters = best_fit_parameters_exp2_dict
        else:
            best_fit_parameters = best_fit_parameters_exp1_dict

        speaker_tau = best_fit_parameters[language][0]
        listener_tau = best_fit_parameters[language][1]

        print('')
        print('')
        print("speaker_tau is:")
        print(speaker_tau)
        print('')
        print("listener_tau is:")
        print(listener_tau)

        if language == "English":
            proximal = "this"
            distal = "that"
        elif language == "Italian":
            proximal = "questo"
            distal = "quello"
        elif language == "Portuguese":
            proximal = "este"
            middle = "esse"
            distal = "aquele"
        elif language == "Spanish":
            proximal = "este"
            middle = "ese"
            distal = "aquel"

        for model in models:
            print('')
            print(model)

            print('')
            print('')
            print(f"LANGUAGE = {language} + MODEL = {model}:")
            # LOAD IN MODEL PREDICTIONS: #
            if experiment == "attention":

                if "attention" in model: #TODO: Get rid of this ad-hoc solution and make it more organised
                    models_for_filename = ["distance_attention", "person_attention"]
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models_for_filename).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')

                else:
                    model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_Attention_'+str(models).replace(" ", "")+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv')
            else:
                model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_' +str(models).replace(" ", "")+'_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv')

            if experiment == "attention":
                if language == "English" or language == "Italian":
                    word_probs_over_situations_dict = {"listener_att":[],
                                                       "origin": [],
                                                       proximal: [],
                                                       distal:[]}
                elif language == "Portuguese" or language == "Spanish":
                    word_probs_over_situations_dict = {"listener_att":[],
                                                       "origin": [],
                                                       proximal: [],
                                                       middle:[],
                                                       distal:[]}

                for i in range(len(object_positions_of_interest)):

                    object_pos = object_positions_of_interest[i]
                    listener_att = listener_attentions_of_interest[i]
                    print('')
                    print('')
                    print("object_pos is:")
                    print(object_pos)
                    print("listener_att is:")
                    print(listener_att)

                    probs_per_word, proportions_per_word = get_probs_and_proportions_for_situation(experiment, model_predictions, data_pd, model, language, speaker_tau, listener_tau, object_pos, listener_pos=None, listener_att=listener_att)
                    if language == "English" or language == "Italian":
                        word_probs_over_situations_dict["listener_att"].append(listener_att)
                        word_probs_over_situations_dict["origin"].append("model predictions")
                        word_probs_over_situations_dict[proximal].append(probs_per_word[0])
                        word_probs_over_situations_dict[distal].append(probs_per_word[1])

                        word_probs_over_situations_dict["listener_att"].append(listener_att)
                        word_probs_over_situations_dict["origin"].append("human participants")
                        word_probs_over_situations_dict[proximal].append(proportions_per_word[0])
                        word_probs_over_situations_dict[distal].append(proportions_per_word[1])
                    elif language == "Portuguese" or language == "Spanish":
                        word_probs_over_situations_dict["listener_att"].append(listener_att)
                        word_probs_over_situations_dict["origin"].append("model predictions")
                        word_probs_over_situations_dict[proximal].append(probs_per_word[0])
                        word_probs_over_situations_dict[middle].append(probs_per_word[1])
                        word_probs_over_situations_dict[distal].append(probs_per_word[2])

                        word_probs_over_situations_dict["listener_att"].append(listener_att)
                        word_probs_over_situations_dict["origin"].append("human participants")
                        word_probs_over_situations_dict[proximal].append(proportions_per_word[0])
                        word_probs_over_situations_dict[middle].append(proportions_per_word[1])
                        word_probs_over_situations_dict[distal].append(proportions_per_word[2])
            else:
                if language == "English" or language == "Italian":
                    word_probs_over_situations_dict = {"listener_pos":[],
                                                       "origin": [],
                                                       proximal: [],
                                                       distal:[]}
                elif language == "Portuguese" or language == "Spanish":
                    word_probs_over_situations_dict = {"listener_pos":[],
                                                       "origin": [],
                                                       proximal: [],
                                                       middle:[],
                                                       distal:[]}

                for i in range(len(object_positions_of_interest)):

                    object_pos = object_positions_of_interest[i]
                    listener_pos = listener_positions_of_interest[i]
                    print('')
                    print('')
                    print("object_pos is:")
                    print(object_pos)
                    print("listener_pos is:")
                    print(listener_pos)

                    probs_per_word, proportions_per_word = get_probs_and_proportions_for_situation(experiment, model_predictions, data_pd, model, language, speaker_tau, listener_tau, object_pos, listener_pos=listener_pos, listener_att=None)
                    if language == "English" or language == "Italian":
                        word_probs_over_situations_dict["listener_pos"].append(listener_pos)
                        word_probs_over_situations_dict["origin"].append("model predictions")
                        word_probs_over_situations_dict[proximal].append(probs_per_word[0])
                        word_probs_over_situations_dict[distal].append(probs_per_word[1])

                        word_probs_over_situations_dict["listener_pos"].append(listener_pos)
                        word_probs_over_situations_dict["origin"].append("human participants")
                        word_probs_over_situations_dict[proximal].append(proportions_per_word[0])
                        word_probs_over_situations_dict[distal].append(proportions_per_word[1])
                    elif language == "Portuguese" or language == "Spanish":
                        word_probs_over_situations_dict["listener_pos"].append(listener_pos)
                        word_probs_over_situations_dict["origin"].append("model predictions")
                        word_probs_over_situations_dict[proximal].append(probs_per_word[0])
                        word_probs_over_situations_dict[middle].append(probs_per_word[1])
                        word_probs_over_situations_dict[distal].append(probs_per_word[2])

                        word_probs_over_situations_dict["listener_pos"].append(listener_pos)
                        word_probs_over_situations_dict["origin"].append("human participants")
                        word_probs_over_situations_dict[proximal].append(proportions_per_word[0])
                        word_probs_over_situations_dict[middle].append(proportions_per_word[1])
                        word_probs_over_situations_dict[distal].append(proportions_per_word[2])


            print('')
            print('')
            print("word_probs_over_situations_dict is:")
            print(word_probs_over_situations_dict)


            pd_word_probs_over_situations = pd.DataFrame.from_dict(word_probs_over_situations_dict)

            pd.set_option('display.max_columns', None)

            print('')
            print('')
            print("pd_word_probs_over_situations is:")
            print(pd_word_probs_over_situations)

            for origin in ["model predictions", "human participants"]:
                if experiment == "attention":
                    plot_stacked_bar_across_model_and_data(experiment, pd_word_probs_over_situations, language, model, origin, object_positions_of_interest, listener_attentions_of_interest=listener_attentions_of_interest)
                else:
                    plot_stacked_bar_across_model_and_data(experiment, pd_word_probs_over_situations, language, model, origin, object_positions_of_interest, listener_positions_of_interest=listener_positions_of_interest)


