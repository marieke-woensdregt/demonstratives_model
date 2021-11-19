import numpy as np
import pandas as pd



# PARAMETER SETTINGS: #
models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]
tau_start = 0.41
tau_stop = 1.41
tau_step = 0.02

tau_start_for_comparison = 0.5
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

    distance_wins_array = distance_wins_array.astype(int)
    person_wins_array = person_wins_array.astype(int)

    for i in range(len(distance_wins_array)):
        for j in range(len(distance_wins_array[i])):
            if distance_wins_array[i][j] == 0 and person_wins_array[i][j] == 0:
                distance_wins_array[i][j] = np.nan
    distance_wins_df = pd.DataFrame(data=distance_wins_array, index=index, columns=column)
    return distance_wins_df




for language in languages:
    print('')
    print('')
    print(language)

    likelihood_df_distance = pd.read_pickle(
        'model_fitting_data/' + 'likelihood_df_' + language + '_' + 'distance' + '_tau_start_' + str(
            tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    print("likelihood_df_distance DISTANCE BEFORE SLICING is:")
    print(likelihood_df_distance)

    likelihood_df_person = pd.read_pickle(
        'model_fitting_data/' + 'likelihood_df_' + language + '_' + 'person' + '_tau_start_' + str(
            tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    print("likelihood_df_person PERSON BEFORE SLICING is:")
    print(likelihood_df_person)


    likelihood_df_distance = likelihood_df_distance[likelihood_df_distance["SpeakerTau"] >= tau_start_for_comparison][likelihood_df_distance["ListenerTau"] >= tau_start_for_comparison]
    # print('')
    # print('')
    # print("likelihood_df_distance AFTER SLICING is:")
    # print(likelihood_df_distance)

    likelihood_df_person = likelihood_df_person[likelihood_df_person["SpeakerTau"] >= tau_start_for_comparison][likelihood_df_person["ListenerTau"] >= tau_start_for_comparison]
    # print('')
    # print('')
    # print("likelihood_df_person AFTER SLICING is:")
    # print(likelihood_df_person)

    likelihood_df_distance = likelihood_df_distance.pivot(index='SpeakerTau', columns='ListenerTau', values='Likelihood')
    # print('')
    # print('')
    # print("likelihood_df_distance AFTER PIVOTING is:")
    # print(likelihood_df_distance)

    likelihood_df_person = likelihood_df_person.pivot(index='SpeakerTau', columns='ListenerTau', values='Likelihood')
    # print('')
    # print('')
    # print("likelihood_df_person AFTER PIVOTING is:")
    # print(likelihood_df_person)

    likelihood_np_array_distance = likelihood_df_distance.to_numpy()
    # print('')
    # print('')
    # print("likelihood_df_distance AFTER CONVERTING TO NUMPY is:")
    # print(likelihood_df_distance)
    # print("likelihood_df_distance.shape is:")
    # print(likelihood_df_distance.shape)
    max_likelihood_distance = np.max(likelihood_np_array_distance)
    print('')
    print('')
    print("max_likelihood_distance is:")
    print(max_likelihood_distance)


    likelihood_np_array_person = likelihood_df_person.to_numpy()
    # print('')
    # print('')
    # print("likelihood_df_person AFTER CONVERTING TO NUMPY is:")
    # print(likelihood_df_person)
    # print("likelihood_df_person.shape is:")
    # print(likelihood_df_person.shape)
    max_likelihood_person = np.max(likelihood_np_array_person)
    print('')
    print('')
    print("max_likelihood_person is:")
    print(max_likelihood_person)


    if language == "English" or language == "Italian":
        bayes_factor_max_likelihood = max_likelihood_distance / max_likelihood_person
        print('')
        print('')
        print("bayes_factor_max_likelihood DISTANCE / PERSON is:")
        print(bayes_factor_max_likelihood)
    elif language == "Portuguese" or language == "Spanish":
        bayes_factor_max_likelihood = max_likelihood_person / max_likelihood_distance
        print('')
        print('')
        print("bayes_factor_max_likelihood PERSON / DISTANCE is:")
        print(bayes_factor_max_likelihood)


    index = likelihood_df_person.index
    # print('')
    # print('')
    # print("index is:")
    # print(index)


    column = likelihood_df_person.columns
    # print('')
    # print('')
    # print("column is:")
    # print(column)


    bayes_factor_array = bayes_factor(likelihood_np_array_distance, likelihood_np_array_person)
    print('')
    print("bayes_factor_array is:")
    print(bayes_factor_array)
    print("bayes_factor_array.shape is:")
    print(bayes_factor_array.shape)


    bayes_factor_df = pd.DataFrame(data=bayes_factor_array, index=index, columns=column)
    print('')
    print("bayes_factor_df is:")
    print(bayes_factor_df)


    distance_wins_df = convert_to_distance_wins(bayes_factor_df)
    print('')
    print('')
    print("distance_wins_df is:")
    print(distance_wins_df)


    bayes_factor_df.to_pickle('model_fitting_data/' + 'bayes_factor_df_' + language + '_' + '_tau_start_' + str(
        tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')


    bayes_factor_df.to_pickle('model_fitting_data/' + 'bayes_factor_df_' + language + '_' + '_tau_start_' + str(
        tau_start_for_comparison) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')