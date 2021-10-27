import pandas as pd
import numpy as np

tau_start = 0.1
tau_stop = 5.4
tau_step = 0.5

n_bins = 20
min_bin_usage = 1

input_file_path = '/Users/U968195/PycharmProjects/demonstratives_model/model_predictions/'
input_file_name = 'HigherSearchD_MW'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv'

input_dataframe = pd.read_csv(input_file_path+input_file_name)
pd.set_option('display.max_columns', None)
print('')
print('')
print(input_dataframe)
print('')
print(input_dataframe.columns)


def get_ranges_per_word(dataframe, model, stau, ltau):
	"""
	Takes a dataframe with model predictions, and returns the cost ranges and probability ranges per word+system combination, for a given speaker_rationality (stau) and listener_rationality (ltau)
	:param dataframe: pandas dataframe containing at least the columns "Model", "SpeakerTau", "ListenerTau", "Word", "WordNo", "Cost", and "Probability"
	:param model: string specifying the model: 'distance', 'person' or 'pdhybrid'
	:param stau: float specifying speaker_rationality
	:param ltau: float specifying listener_rationality
	:return: two dictionaries: (1) a dictionary containing a numpy array of cost values for each word+system combination; (2) a dictionary containing a numpy array of probabilities for each word+system combination
	"""
	prob_range_dict = {"este_two":np.array([]),
	"aquel_two":np.array([]),
	"este_three":np.array([]),
	"ese_three":np.array([]),
	"aquel_three":np.array([])}
	cost_range_dict = {"este_two":np.array([]),
	"aquel_two":np.array([]),
	"este_three":np.array([]),
	"ese_three":np.array([]),
	"aquel_three":np.array([])}
	word_system_combos = [["este", 2], ["aquel", 2], ["este", 3], ["ese", 3], ["aquel", 3]]
	for combo in word_system_combos:
		word = combo[0]
		system = combo[1]
		relevant_rows = dataframe[dataframe["Model"] == model][dataframe["SpeakerTau"] == round(stau, 2)][dataframe["ListenerTau"] == round(ltau, 2)][dataframe["Word"] == word][dataframe["WordNo"] == system]
		costs = relevant_rows["Cost"]
		costs_array = costs.to_numpy()
		probabilities = relevant_rows["Probability"]
		probabilities_array = probabilities.to_numpy()
		if word == "este":
			if system == 2:
				prob_range_dict["este_two"] = probabilities_array
				cost_range_dict["este_two"] = costs_array
			elif system == 3:
				prob_range_dict["este_three"] = probabilities_array
				cost_range_dict["este_three"] = costs_array
		elif word == "ese" and system == 3:
			prob_range_dict["ese_three"] = probabilities_array
			cost_range_dict["ese_three"] = costs_array
		elif word == "aquel":
			if system == 2:
				prob_range_dict["aquel_two"] = probabilities_array
				cost_range_dict["aquel_two"] = costs_array
			elif system == 3:
				prob_range_dict["aquel_three"] = probabilities_array
				cost_range_dict["aquel_three"] = costs_array
	return prob_range_dict, cost_range_dict


def bincount_ranges(range_dict, range_min, range_max, n_bins, min_bin_usage):
	n_bins_used_dict = {}
	for combo_name, probabilities in range_dict.items():
		histogram_values = np.histogram(probabilities, bins=n_bins, range=(range_min, range_max))
		bins_used = np.where(histogram_values[0] >= min_bin_usage)[0]
		n_bins_used = len(bins_used)
		n_bins_used_dict[combo_name] = [histogram_values, n_bins_used]
	return n_bins_used_dict


def turn_bins_used_dict_into_spread_dict(input_df, measure, range_min, range_max, bin_size):
	spread_dict ={"Model":[],
					   "ltau":[],
					   "stau":[],
					   "Word":[],
					   "WordNo":[],
					   "n_bins_used":[]}
	for listener_rationality in np.arange(tau_start, tau_stop, tau_step):
		for speaker_rationality in np.arange(tau_start, tau_stop, tau_step):
			prob_range_dict, cost_range_dict = get_ranges_per_word(input_df, model, speaker_rationality, listener_rationality)
			if measure == "Cost":
				n_bins_used_dict = bincount_ranges(cost_range_dict, range_min, range_max, n_bins, min_bin_usage)
			elif measure == "Probability":
				n_bins_used_dict = bincount_ranges(prob_range_dict, range_min, range_max, n_bins, min_bin_usage)
			for key, value in n_bins_used_dict.items():
				spread_dict["Model"].append(model)
				spread_dict["ltau"].append(round(listener_rationality, 2))
				spread_dict["stau"].append(round(speaker_rationality, 2))
				if key == "este_two":
					spread_dict["Word"].append("este")
					spread_dict["WordNo"].append(2)
				elif key == "aquel_two":
					spread_dict["Word"].append("aquel")
					spread_dict["WordNo"].append(2)
				elif key == "este_three":
					spread_dict["Word"].append("este")
					spread_dict["WordNo"].append(3)
				elif key == "ese_three":
					spread_dict["Word"].append("ese")
					spread_dict["WordNo"].append(3)
				elif key == "aquel_three":
					spread_dict["Word"].append("aquel")
					spread_dict["WordNo"].append(3)
				spread_dict["n_bins_used"].append(value[1])
				for i in range(len(value[0][0])):
					bin_name = str(round(value[0][1][i], 2))+"-"+str(round((value[0][1][i] + bin_size), 2))
					if bin_name in spread_dict.keys():
						spread_dict[bin_name].append(value[0][0][i])
					else:
						spread_dict[bin_name] = [value[0][0][i]]
	return spread_dict


def save_spread_df_to_csv(model, spread_df, output_file_path, output_file_name):
	pd.set_option('display.max_columns', None)
	# if model == "distance":
	# 	max_n_bins = np.max(spread_df[spread_df["WordNo"] == 2]["n_bins_used"].to_numpy())
	# 	print("max_n_bins is:")
	# 	print(max_n_bins)
	# 	print(spread_df[spread_df["WordNo"] == 2][spread_df["n_bins_used"] == max_n_bins])
	# elif model == "person":
	# 	max_n_bins = np.max(spread_df[spread_df["WordNo"] == 3]["n_bins_used"].to_numpy())
	# 	print("max_n_bins is:")
	# 	print(max_n_bins)
	# 	print(spread_df[spread_df["WordNo"] == 3][spread_df["n_bins_used"] == max_n_bins])
	if model == "distance":
		sorted_df = spread_df[spread_df["WordNo"] == 2].sort_values(by="n_bins_used")
		sorted_df.to_csv(output_file_path + output_file_name, index=False)
	elif model == "person":
		sorted_df = spread_df[spread_df["WordNo"] == 3].sort_values(by="n_bins_used")
		sorted_df.to_csv(output_file_path + output_file_name, index=False)



for model in ['distance', 'person']:
	print('')
	print('')
	print(f"MODEL IS: {model}:")

	bin_size_prob = 1.0 / n_bins
	bin_size_cost = 3.0 / n_bins

	cost_spread_dict = turn_bins_used_dict_into_spread_dict(input_dataframe, "Cost", -3.0, -0.0, bin_size_cost)
	prob_spread_dict = turn_bins_used_dict_into_spread_dict(input_dataframe, "Probability", 0.0, 1.0, bin_size_prob)

	cost_spread_df = pd.DataFrame(data=cost_spread_dict)
	prob_spread_df = pd.DataFrame(data=prob_spread_dict)

	output_file_path = '/Users/U968195/PycharmProjects/demonstratives_model/model_predictions/'
	output_file_name = model + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '.csv'
	save_spread_df_to_csv(model, cost_spread_df, output_file_path, 'Cost_Spread_' + output_file_name)
	save_spread_df_to_csv(model, prob_spread_df, output_file_path, 'Prob_Spread_' + output_file_name)



