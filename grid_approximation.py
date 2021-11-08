import pandas as pd
import numpy as np
from scipy.stats import binom, multinomial
import seaborn as sns


# LOAD IN DATA: #
data_exp_1_two_system = pd.read_csv('data/with_counts/TwoSystem.csv', index_col=0)
data_exp_1_three_system = pd.read_csv('data/with_counts/ThreeSystem.csv', index_col=0)


# LOAD IN MODEL PREDICTIONS: #
model_predictions = pd.read_csv('model_predictions/HigherSearchD_MW_RSA_tau_start_0.1_tau_stop_0.61_tau_step_0.05.csv')


# PARAMETER SETTINGS: #
model = "distance"  # can be set to either "distance" or "person"
language = "English"  # can be set to "English", "Italian", "Portuguese" or "Spanish"
object_positions = [0, 1, 2, 3]  # array of all possible object (= referent) positions
listener_positions = [0, 1, 2, 3]  # array of all possible listener positions
speaker_tau = 0.4
listener_tau = 0.4


# FUNCTION DEFINITIONS: #

def calc_multinom_pmf(pd_model_predictions, pd_data, model, language, object_pos, listener_pos, speaker_tau, listener_tau):
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
        print('')
        print("word is:")
        print(word)
        model_prediction_row = pd_model_predictions[model_predictions["Model"]==model][model_predictions["Word"]==word][model_predictions["Referent"]==object_pos][model_predictions["Listener_pos"]==listener_pos][model_predictions["WordNo"]==WordNo][model_predictions["SpeakerTau"]==speaker_tau][model_predictions["ListenerTau"]==listener_tau]
        print("model_prediction_row is:")
        print(model_prediction_row)
        model_prediction_prob = model_prediction_row["Probability"]
        print("model_prediction_prob is:")
        print(model_prediction_prob)
        probs_per_word[i] = model_prediction_prob
        # Below is object_pos+1 and listener_pos+1, because in the model_predictions dataframe it starts counting
        # from 0, but in the experimental data dataframe it starts counting from 1.
        data_count_row = pd_data[data_exp_1_two_system["Object_Position"] == object_pos+1][data_exp_1_two_system["Listener_Position"] == listener_pos+1][data_exp_1_two_system["Language"] == language]
        print('')
        print("data_count_row is:")
        print(data_count_row)
        data_count = data_count_row[word]
        print("data_count is:")
        print(data_count)
        counts_per_word[i] = data_count
        total = data_count_row["Total"]
        print("total is:")
        print(total)
    multinom_pmf = multinomial.pmf(counts_per_word, n=total, p=probs_per_word)
    multinom_logpmf = multinomial.logpmf(counts_per_word, n=total, p=probs_per_word)
    return multinom_pmf, multinom_logpmf


def product_logpmf_over_situations(object_positions, listener_positions, pd_model_predictions, pd_data, model, language, speaker_tau, listener_tau):
    logproduct = np.log(1.0)  # The first probability should be multiplied with 1.0, which is equivalent to 0.0 in log-space
    for object_pos in object_positions:
        print('')
        print('')
        print("object_pos is:")
        print(object_pos)
        for listener_pos in listener_positions:
            print('')
            print("listener_pos is:")
            print(listener_pos)
            multinom_pmf, multinom_logpmf = calc_multinom_pmf(pd_model_predictions, pd_data, model, language, object_pos, listener_pos, speaker_tau, listener_tau)
            print("multinom_pmf is:")
            print(multinom_pmf)
            print("multinom_logpmf is:")
            print(multinom_logpmf)
            print("np.exp(multinom_logpmf) is:")
            print(np.exp(multinom_logpmf))
            logproduct += multinom_logpmf  # multiplication in probability space is equivalent to addition in log-space
    return logproduct


logproduct = product_logpmf_over_situations(object_positions, listener_positions, model_predictions, data_exp_1_two_system, model, language, speaker_tau, listener_tau)
print('')
print('')
print("logproduct is:")
print(logproduct)
print("np.exp(logproduct) is:")
print(np.exp(logproduct))