import numpy as np
import pandas as pd
import pickle5 as p
import pickle


# PARAMETER SETTINGS: #

rsa_layer = False  # Can be set to either True or False

ese_uniform = True  # Can be set to either True or False. Determines whether "ese" under the simple distance model is a uniform distribution (if set to True), or rather centred around the medial objects (if set to False)

comparison = "rsa_contribution"  # Can be set to either "system", "attention_correction" or "rsa_contribution"

experiment = "attention"  # Can be set to either "baseline" (Experiment 1) or "attention" (Experiment 2)

if experiment == "attention":
    models = ["distance_attention", "person_attention"]
else:
    models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]
tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

tau_start_for_comparison = 0.4
# tau_stop_for_comparison = 1.41


def bayes_factor(likelihood_np_array_distance, likelihood_np_array_person):
    bayes_factor_array = np.divide(likelihood_np_array_distance, likelihood_np_array_person)
    return bayes_factor_array



def convert_to_distance_wins(bayes_factor_df):
    index = bayes_factor_df.index
    column = bayes_factor_df.columns
    bayes_factor_array = bayes_factor_df.to_numpy()

    distance_wins_array = np.array(bayes_factor_array > 1)
    person_wins_array = np.array(bayes_factor_array < 1)

    distance_wins_array = distance_wins_array.astype(int)  # to convert from True/False to 1/0
    person_wins_array = person_wins_array.astype(int)  # to convert from True/False to 1/0

    distance_wins_array = distance_wins_array.astype(float)  # to convert from int to float
    person_wins_array = person_wins_array.astype(float)  # to convert from int to float

    for i in range(len(distance_wins_array)):
        for j in range(len(distance_wins_array[i])):
            if distance_wins_array[i][j] == 0 and person_wins_array[i][j] == 0:
                distance_wins_array[i][j] = np.nan
    distance_wins_df = pd.DataFrame(data=distance_wins_array, index=index, columns=column)
    return distance_wins_df


def convert_to_strength_of_evidence(bayes_factor_df):
    index = bayes_factor_df.index
    column = bayes_factor_df.columns
    evidence_strength_array = bayes_factor_df.to_numpy()

    evidence_strength_array[(evidence_strength_array == 1.)] = np.nan
    evidence_strength_array[(((evidence_strength_array < 1.) & (evidence_strength_array > (1. / 3.))) |
                    ((evidence_strength_array > 1.) & (evidence_strength_array < 3.)))] = 1
    evidence_strength_array[(((evidence_strength_array < (1. / 3.)) & (evidence_strength_array > (1. / 10.))) |
                    ((evidence_strength_array > 3.) & (evidence_strength_array < 10.)))] = 2
    evidence_strength_array[(((evidence_strength_array < (1. / 10.)) & (evidence_strength_array > (1. / 30.))) |
                    ((evidence_strength_array > 10.) & (evidence_strength_array < 30.)))] = 3
    evidence_strength_array[(((evidence_strength_array < (1./30.)) & (evidence_strength_array > (1./100.))) |
                    ((evidence_strength_array > 30.) & (evidence_strength_array < 100.)))] = 4
    evidence_strength_array[((evidence_strength_array < (1./100.)) |
                    (evidence_strength_array > 100))] = 5
    print('')
    print('')
    print("evidence_strength_array is:")
    print(evidence_strength_array)


    # distance_wins_array = np.array(evidence_strength_array > 1)
    # person_wins_array = np.array(evidence_strength_array < 1)
    #
    # distance_wins_array = distance_wins_array.astype(int)  # to convert from True/False to 1/0
    # person_wins_array = person_wins_array.astype(int)  # to convert from True/False to 1/0
    #
    # distance_wins_array = distance_wins_array.astype(float)  # to convert from int to float
    # person_wins_array = person_wins_array.astype(float)  # to convert from int to float
    #
    # for i in range(len(distance_wins_array)):
    #     for j in range(len(distance_wins_array[i])):
    #         if distance_wins_array[i][j] == 0 and person_wins_array[i][j] == 0:
    #             distance_wins_array[i][j] = np.nan

    evidence_strength_df = pd.DataFrame(data=evidence_strength_array, index=index, columns=column)

    return evidence_strength_df


for language in languages:
    print('')
    print('')
    print('')
    print('')
    print(language)


    if comparison == "system":
        with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + 'distance' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
            likelihood_df_baseline = p.load(fh)
            print('')
            print("likelihood_df_baseline is:")
            print(likelihood_df_baseline)
        # likelihood_df_baseline = pd.read_pickle(
        #     'model_fitting_data/' + 'likelihood_df_' + language + '_' + 'distance' + '_tau_start_' + str(
        #         tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        # # print("likelihood_df_distance DISTANCE BEFORE SLICING is:")
        # # print(likelihood_df_distance)

        with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + 'person' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
            likelihood_df_comparison = p.load(fh)
            print('')
            print("likelihood_df_comparison is:")
            print(likelihood_df_comparison)
        # likelihood_df_comparison = pd.read_pickle(
        #     'model_fitting_data/' + 'likelihood_df_' + language + '_' + 'person' + '_tau_start_' + str(
        #         tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
        # # print("likelihood_df_person PERSON BEFORE SLICING is:")
        # # print(likelihood_df_person)


    elif comparison == "attention_correction":

        if language == 'English' or language == 'Italian':

            with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + 'distance' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df_baseline = p.load(fh)
                print("likelihood_df_baseline DISTANCE BEFORE SLICING is:")
                print(likelihood_df_baseline)
            # likelihood_df_baseline = pd.read_pickle(
            #     'model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_' + 'distance' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
            # print("likelihood_df_baseline DISTANCE BEFORE SLICING is:")
            # print(likelihood_df_baseline)

            with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + 'distance_attention' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df_comparison = p.load(fh)
                print("likelihood_df_comparison DISTANCE + ATTENTION BEFORE SLICING is:")
                print(likelihood_df_comparison)
            # likelihood_df_comparison = pd.read_pickle(
            #     'model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + 'distance_attention' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
            # print("likelihood_df_comparison DISTANCE + ATTENTION BEFORE SLICING is:")
            # print(likelihood_df_comparison)

        if language == 'Spanish' or language == 'Portuguese':
            with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + 'person' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df_baseline = p.load(fh)
                print("likelihood_df_baseline PERSON BEFORE SLICING is:")
                print(likelihood_df_baseline)
            # likelihood_df_baseline = pd.read_pickle(
            #     'model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + 'person' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
            # print("likelihood_df_baseline PERSON BEFORE SLICING is:")
            # print(likelihood_df_baseline)

            with open('model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + 'person_attention' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df_comparison = p.load(fh)
                print("likelihood_df_comparison PERSON + ATTENTION BEFORE SLICING is:")
                print(likelihood_df_comparison)
            # likelihood_df_comparison = pd.read_pickle(
            #     'model_fitting_data/' + 'likelihood_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + 'person_attention' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
            # print("likelihood_df_comparison PERSON + ATTENTION BEFORE SLICING is:")
            # print(likelihood_df_comparison)


    elif comparison == "rsa_contribution":

        if language == 'English' or language == 'Italian':
            model = 'distance'
        elif language == 'Spanish' or language == 'Portuguese':
            model = 'person'

        if experiment == "baseline":
                with open('model_fitting_data/' + 'likelihood_df_RSA_False_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                    likelihood_df_baseline = p.load(fh)
                    print("likelihood_df_baseline RSA FALSE is:")
                    print(likelihood_df_baseline)

                with open('model_fitting_data/' + 'likelihood_df_RSA_True_Ese_uniform_True_' + language + '_' + model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                    likelihood_df_comparison = p.load(fh)
                    print("likelihood_df_comparison RSA TRUE is:")
                    print(likelihood_df_comparison)

        elif experiment == "attention":

            with open('model_fitting_data/' + 'likelihood_df_RSA_False_Attention_Ese_uniform_' + str(ese_uniform) + '_' +  language + '_' + model + '_attention' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df_baseline = p.load(fh)
                print("likelihood_df_baseline RSA FALSE is:")
                print(likelihood_df_baseline)

            with open('model_fitting_data/' + 'likelihood_df_RSA_True_Attention_Ese_uniform_True_' + language + '_' + model + '_attention' + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl', "rb") as fh:
                likelihood_df_comparison = p.load(fh)
                print("likelihood_df_comparison RSA TRUE is:")
                print(likelihood_df_comparison)


    likelihood_df_baseline = likelihood_df_baseline[likelihood_df_baseline["SpeakerTau"] >= tau_start_for_comparison][likelihood_df_baseline["ListenerTau"] >= tau_start_for_comparison]
    print('')
    print('')
    print("likelihood_df_baseline AFTER SLICING is:")
    print(likelihood_df_baseline)

    likelihood_df_comparison = likelihood_df_comparison[likelihood_df_comparison["SpeakerTau"] >= tau_start_for_comparison][likelihood_df_comparison["ListenerTau"] >= tau_start_for_comparison]
    # print('')
    # print('')
    # print("likelihood_df_person AFTER SLICING is:")
    # print(likelihood_df_person)

    likelihood_df_baseline = likelihood_df_baseline.pivot(index='SpeakerTau', columns='ListenerTau', values='Likelihood')
    # print('')
    # print('')
    # print("likelihood_df_distance AFTER PIVOTING is:")
    # print(likelihood_df_distance)

    likelihood_df_comparison = likelihood_df_comparison.pivot(index='SpeakerTau', columns='ListenerTau', values='Likelihood')
    # print('')
    # print('')
    # print("likelihood_df_person AFTER PIVOTING is:")
    # print(likelihood_df_person)


    index_baseline = likelihood_df_baseline.index
    column_baseline = likelihood_df_baseline.columns
    # print('')
    # print('')
    # print("index_distance is:")
    # print(index_distance)
    # print("len(index_distance) is:")
    # print(len(index_distance))
    # print('')
    # print("column_distance is:")
    # print(column_distance)
    # print("len(column_distance) is:")
    # print(len(column_distance))

    index_comparison = likelihood_df_comparison.index
    column_comparison = likelihood_df_comparison.columns
    # print('')
    # print('')
    # print("index_person is:")
    # print(index_person)
    # print("len(index_person) is:")
    # print(len(index_person))
    # print('')
    # print("column_person is:")
    # print(column_person)
    # print("len(column_person) is:")
    # print(len(column_person))


    likelihood_np_array_baseline = likelihood_df_baseline.to_numpy()
    print('')
    print('')
    print("likelihood_np_array_baseline AFTER CONVERTING TO NUMPY is:")
    print(likelihood_np_array_baseline)
    print("likelihood_np_array_baseline.shape is:")
    print(likelihood_np_array_baseline.shape)
    # max_likelihood_baseline = np.amax(likelihood_np_array_baseline)
    max_likelihood_baseline = np.nanmax(likelihood_np_array_baseline)
    print('')
    print('')
    print("max_likelihood_baseline is:")
    print(max_likelihood_baseline)
    max_index_likelihood_baseline = np.where(likelihood_np_array_baseline == max_likelihood_baseline)
    print('')
    print('')
    print("max_index_likelihood_baseline is:")
    print(max_index_likelihood_baseline)


    likelihood_np_array_comparison = likelihood_df_comparison.to_numpy()
    print('')
    print('')
    print("likelihood_np_array_comparison AFTER CONVERTING TO NUMPY is:")
    print(likelihood_np_array_comparison)
    print("likelihood_np_array_comparison.shape is:")
    print(likelihood_np_array_comparison.shape)
    # max_likelihood_comparison = np.amax(likelihood_np_array_comparison)
    max_likelihood_comparison = np.nanmax(likelihood_np_array_comparison)
    print('')
    print('')
    print("max_likelihood_comparison is:")
    print(max_likelihood_comparison)
    max_index_likelihood_comparison = np.where(likelihood_np_array_comparison == max_likelihood_comparison)
    print('')
    print('')
    print("max_index_likelihood_comparison is:")
    print(max_index_likelihood_comparison)


    if comparison == "system":
        if language == "English" or language == "Italian":
            bayes_factor_max_likelihood = max_likelihood_baseline / max_likelihood_comparison
            print('')
            print('')
            print("bayes_factor_max_likelihood DISTANCE / PERSON is:")
            print(bayes_factor_max_likelihood)
            max_likelihood_coordinates = max_index_likelihood_baseline
            print('')
            print("max_likelihood_coordinates are:")
            print(max_likelihood_coordinates)
            speaker_tau_max_likelihood = index_baseline[max_likelihood_coordinates[0][0]]
            print('')
            print("speaker_tau_max_likelihood is:")
            print(speaker_tau_max_likelihood)
            listener_tau_max_likelihood = column_baseline[max_likelihood_coordinates[1][0]]
            print('')
            print("listener_tau_max_likelihood is:")
            print(listener_tau_max_likelihood)

        elif language == "Portuguese" or language == "Spanish":
            bayes_factor_max_likelihood = max_likelihood_comparison / max_likelihood_baseline
            print('')
            print('')
            print("bayes_factor_max_likelihood PERSON / DISTANCE is:")
            print(bayes_factor_max_likelihood)
            max_likelihood_coordinates = max_index_likelihood_comparison
            print('')
            print("max_likelihood_coordinates are:")
            print(max_likelihood_coordinates)
            speaker_tau_max_likelihood = index_comparison[max_likelihood_coordinates[0][0]]
            print('')
            print("speaker_tau_max_likelihood is:")
            print(speaker_tau_max_likelihood)
            listener_tau_max_likelihood = column_comparison[max_likelihood_coordinates[1][0]]
            print('')
            print("listener_tau_max_likelihood is:")
            print(listener_tau_max_likelihood)

    else:  # i.e. if comparison == "attention_correction" OR comparison == "rsa_contribution"
        bayes_factor_max_likelihood = max_likelihood_comparison / max_likelihood_baseline
        print('')
        print('')
        print("bayes_factor_max_likelihood COMPARISON / BASELINE is:")
        print(bayes_factor_max_likelihood)

        max_likelihood_coordinates_baseline = max_index_likelihood_baseline
        print('')
        print("max_likelihood_coordinates_baseline BASELINE are:")
        print(max_likelihood_coordinates_baseline)
        speaker_tau_max_likelihood_baseline = index_baseline[max_likelihood_coordinates_baseline[0][0]]
        print('')
        print("speaker_tau_max_likelihood_baseline BASELINE is:")
        print(speaker_tau_max_likelihood_baseline)
        listener_tau_max_likelihood_baseline = column_baseline[max_likelihood_coordinates_baseline[1][0]]
        print('')
        print("listener_tau_max_likelihood_baseline BASELINE is:")
        print(listener_tau_max_likelihood_baseline)


        max_likelihood_coordinates_comparison = max_index_likelihood_comparison
        print('')
        print("max_likelihood_coordinates_comparison COMPARISON are:")
        print(max_likelihood_coordinates_comparison)
        speaker_tau_max_likelihood_comparison = index_comparison[max_likelihood_coordinates_comparison[0][0]]
        print('')
        print("speaker_tau_max_likelihood_comparison COMPARISON is:")
        print(speaker_tau_max_likelihood_comparison)
        listener_tau_max_likelihood_comparison = column_comparison[max_likelihood_coordinates_comparison[1][0]]
        print('')
        print("listener_tau_max_likelihood_comparison COMPARISON is:")
        print(listener_tau_max_likelihood_comparison)


    index = likelihood_df_comparison.index
    # print('')
    # print('')
    # print("index is:")
    # print(index)


    column = likelihood_df_comparison.columns
    # print('')
    # print('')
    # print("column is:")
    # print(column)


    if comparison == "system":

        bayes_factor_array = bayes_factor(likelihood_np_array_baseline, likelihood_np_array_comparison)
        # print('')
        # print("bayes_factor_array is:")
        # print(bayes_factor_array)
        # print("bayes_factor_array.shape is:")
        # print(bayes_factor_array.shape)

        bayes_factor_df = pd.DataFrame(data=bayes_factor_array, index=index, columns=column)
        print('')
        print('')
        print("bayes_factor_df is:")
        print(bayes_factor_df)

        distance_wins_df = convert_to_distance_wins(bayes_factor_df)
        print('')
        print('')
        print("distance_wins_df is:")
        print(distance_wins_df)

        evidence_strength_df = convert_to_strength_of_evidence(bayes_factor_df)
        print('')
        print('')
        print("evidence_strength_df is:")
        print(evidence_strength_df)

        bayes_factor_df.to_pickle('model_fitting_data/' + 'bayes_factor_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

        distance_wins_df.to_pickle('model_fitting_data/' + 'distance_wins_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

        evidence_strength_df.to_pickle('model_fitting_data/' + 'evidence_strength_df_RSA_'+str(rsa_layer)+'_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')


    elif comparison == "attention_correction":

        bayes_factor_array = bayes_factor(likelihood_np_array_comparison, likelihood_np_array_baseline)
        # print('')
        # print("bayes_factor_array is:")
        # print(bayes_factor_array)
        # print("bayes_factor_array.shape is:")
        # print(bayes_factor_array.shape)

        bayes_factor_df = pd.DataFrame(data=bayes_factor_array, index=index, columns=column)
        print('')
        print('')
        print("bayes_factor_df is:")
        print(bayes_factor_df)

        attention_wins_df = convert_to_distance_wins(bayes_factor_df)
        print('')
        print('')
        print("attention_wins_df is:")
        print(attention_wins_df)

        evidence_strength_df = convert_to_strength_of_evidence(bayes_factor_df)
        print('')
        print('')
        print("evidence_strength_df is:")
        print(evidence_strength_df)

        bayes_factor_df.to_pickle('model_fitting_data/' + 'bayes_factor_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

        attention_wins_df.to_pickle('model_fitting_data/' + 'attention_wins_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

        evidence_strength_df.to_pickle('model_fitting_data/' + 'evidence_strength_df_RSA_'+str(rsa_layer)+'_Attention_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(
            tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')


    elif comparison == "rsa_contribution":

        bayes_factor_array = bayes_factor(likelihood_np_array_comparison, likelihood_np_array_baseline)
        # print('')
        # print("bayes_factor_array is:")
        # print(bayes_factor_array)
        # print("bayes_factor_array.shape is:")
        # print(bayes_factor_array.shape)

        bayes_factor_df = pd.DataFrame(data=bayes_factor_array, index=index, columns=column)
        print('')
        print('')
        print("bayes_factor_df is:")
        print(bayes_factor_df)

        rsa_wins_df = convert_to_distance_wins(bayes_factor_df)
        print('')
        print('')
        print("rsa_wins_df is:")
        print(rsa_wins_df)

        evidence_strength_df = convert_to_strength_of_evidence(bayes_factor_df)
        print('')
        print('')
        print("evidence_strength_df is:")
        print(evidence_strength_df)

        if experiment == "baseline":

            bayes_factor_df.to_pickle('model_fitting_data/' + 'bayes_factor_df_RSA_Contribution_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

            rsa_wins_df.to_pickle('model_fitting_data/' + 'RSA_wins_df_RSA_Contribution_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

            evidence_strength_df.to_pickle('model_fitting_data/' + 'evidence_strength_df_RSA_Contribution_Ese_uniform_' + str(ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

        elif experiment == "attention":

            bayes_factor_df.to_pickle('model_fitting_data/' + 'bayes_factor_df_RSA_Contribution_Attention_Ese_uniform_' + str(
                ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(
                tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

            rsa_wins_df.to_pickle('model_fitting_data/' + 'RSA_wins_df_RSA_Contribution_Attention_Ese_uniform_' + str(
                ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(
                tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

            evidence_strength_df.to_pickle(
                'model_fitting_data/' + 'evidence_strength_df_RSA_Contribution_Attention_Ese_uniform_' + str(
                    ese_uniform) + '_' + language + '_tau_start_' + str(tau_start_for_comparison) + '_tau_stop_' + str(
                    tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')

