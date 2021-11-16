import numpy as np
import pandas as pd



# PARAMETER SETTINGS: #
models = ["distance", "person"]
languages = ["English", "Italian", "Portuguese", "Spanish"]
tau_start = 0.41
tau_stop = 1.41
tau_step = 0.02

tau_start_for_plot = 0.5
# tau_stop_for_plot = 1.41


def likelihood_ratio(likelihood_np_array_distance, likelihood_np_array_person):
    bayes_factor_array = np.divide(likelihood_np_array_distance, likelihood_np_array_person)
    return bayes_factor_array




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
        'model_fitting_data/' + 'likelihood_df_' + language + '_' + 'distance' + '_tau_start_' + str(
            tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.pkl')
    print("likelihood_df_person PERSON BEFORE SLICING is:")
    print(likelihood_df_person)


    likelihood_df_distance = likelihood_df_distance[likelihood_df_distance["SpeakerTau"] >= tau_start_for_plot][likelihood_df_distance["ListenerTau"] >= tau_start_for_plot]
    print('')
    print('')
    print("likelihood_df_distance AFTER SLICING is:")
    print(likelihood_df_distance)

    likelihood_df_person = likelihood_df_person[likelihood_df_person["SpeakerTau"] >= tau_start_for_plot][likelihood_df_person["ListenerTau"] >= tau_start_for_plot]
    print('')
    print('')
    print("likelihood_df_person AFTER SLICING is:")
    print(likelihood_df_person)

    likelihood_df_distance = likelihood_df_distance.pivot(index='SpeakerTau', columns='ListenerTau', values='LogLikelihood')
    print('')
    print('')
    print("likelihood_df_distance AFTER PIVOTING is:")
    print(likelihood_df_distance)

    likelihood_df_person = likelihood_df_person.pivot(index='SpeakerTau', columns='ListenerTau', values='LogLikelihood')
    print('')
    print('')
    print("likelihood_df_person AFTER PIVOTING is:")
    print(likelihood_df_person)

    likelihood_np_array_distance = likelihood_df_distance.to_numpy()
    print('')
    print('')
    print("likelihood_df_distance AFTER CONVERTING TO NUMPY is:")
    print(likelihood_df_distance)
    print("likelihood_df_distance.shape is:")
    print(likelihood_df_distance.shape)

    likelihood_np_array_person = likelihood_df_person.to_numpy()
    print('')
    print('')
    print("likelihood_df_person AFTER CONVERTING TO NUMPY is:")
    print(likelihood_df_person)
    print("likelihood_df_person.shape is:")
    print(likelihood_df_person.shape)


    bayes_factor_array = likelihood_ratio(likelihood_np_array_distance, likelihood_np_array_person)
    print('')
    print('')
    print("bayes_factor_array is:")
    print(bayes_factor_array)
    print("bayes_factor_array.shape is:")
    print(bayes_factor_array.shape)