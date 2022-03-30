import LiteralSpeaker_MW
import PragmaticListener
import PragmaticSpeaker_MW
import numpy as np


########################################################################################
# PARAMETER SETTINGS:

models = ['distance', 'person']  # Can contain: 'distance','person','pdhybrid', 'distance_attention', 'person_attention'

listener_positions = [0, 1, 2, 3]
object_positions = [0, 1, 2, 3]

speaker_rationality = 0.45
listener_rationality = 0.90

output_dict = {"Model":[],
			   "Word":[],
			   # "Cost":[],
			   "Probability":[],
			   "Referent":[],
			   "Speaker_pos":[],
			   "Listener_pos":[],
			   "Listener_att":[],
			   "WordNo":[],
			   "SpeakerTau":[],
			   "ListenerTau":[]}
########################################################################################



def utility_proximal(model, referent_position, speaker_position):
	if model == "distance" or model == "person":
		utility = -np.abs(referent_position - speaker_position)
	return utility


def utility_medial(model, referent_position, speaker_position, listener_position, n_referents):
	if model == "distance":
		utility = 1./n_referents
	elif model == "person":
		distance_from_speaker = np.abs(referent_position - speaker_position)
		distance_from_listener = np.abs(referent_position - listener_position)
		utility = distance_from_speaker + (-distance_from_listener)
	return utility


def utility_distal(model, referent_position, speaker_position, listener_position):
	if model == "distance":
		utility = np.abs(referent_position - speaker_position)
	elif model == "person":
		distance_from_speaker = np.abs(referent_position - speaker_position)
		distance_from_listener = np.abs(referent_position - listener_position)
		utility = distance_from_speaker + distance_from_listener
	return utility


def utilities_across_referents(model, speaker_position, listener_position, term, object_positions):
	utilities_array = np.zeros(len(object_positions))
	for i in range(len(object_positions)):
		referent_position = object_positions[i]
		if term == "proximal":
			utility = utility_proximal(model, referent_position, speaker_position)
		elif term == "medial":
			utility = utility_medial(model, referent_position, speaker_position, listener_position, len(object_positions))
		elif term == "distal":
			utility = utility_distal(model, referent_position, speaker_position, listener_position)
		utilities_array[i] = utility
	return utilities_array


def scale_utilities(distance_values):
	utilities_scaled = np.subtract(distance_values, np.amin(distance_values))
	if np.sum(utilities_scaled) == 0:
		utilities_scaled = np.array([1./len(utilities_scaled) for x in range(len(utilities_scaled))])
	else:
		utilities_scaled = np.divide(utilities_scaled, np.amax(utilities_scaled))
	return utilities_scaled


# def normalize_utilities(utilities_scaled):
# 	if sum(utilities_scaled) == 0:
# 		utilities_normalized_julian = [1.0 / len(utilities_scaled)] * len(utilities_scaled)
# 		# print("utilities_normalized_julian JULIAN'S METHOD:")
# 		# print(utilities_normalized_julian)
# 		julians_utilities_normalized = np.divide(utilities_normalized_julian, np.sum(utilities_normalized_julian))
# 		# print("julians_utilities_normalized are:")
# 		# print(julians_utilities_normalized)
# 		utilities_normalized = [1./len(utilities_scaled) for x in range(len(utilities_scaled))]
# 		# print("utilities_normalized MY METHOD:")
# 		# print(utilities_normalized)
# 	else:
# 		utilities_normalized_julian = [x * 1.0 / max(utilities_scaled) for x in utilities_scaled]
# 		# print("utilities_normalized_julian JULIAN'S METHOD:")
# 		# print(utilities_normalized_julian)
# 		julians_utilities_normalized = np.divide(utilities_normalized_julian, np.sum(utilities_normalized_julian))
# 		# print("julians_utilities_normalized are:")
# 		# print(julians_utilities_normalized)
# 		utilities_scaled_as_array = np.array(utilities_scaled)
# 		utilities_normalized = np.divide(utilities_scaled_as_array, np.sum(utilities_scaled_as_array))
# 		# print("utilities_normalized MY METHOD:")
# 		# print(utilities_normalized)
# 	return utilities_normalized_julian, utilities_normalized


def normalize_search_costs(search_costs):
	search_costs_scaled = np.array([x + 3 for x in search_costs])  # shift by 3 so we're on a 0-3 scale
	return search_costs_scaled


def fixation_probs(utilities_normalized, listener_tau):
	if sum(utilities_normalized) == 0:
		utilities_softmaxed = np.array([1./len(utilities_normalized) for x in range(len(utilities_normalized))])
	else:
		utilities_normalized = np.array(utilities_normalized)
		utilities_softmaxed = np.exp(utilities_normalized/listener_tau)
		utilities_softmaxed = np.divide(utilities_softmaxed, np.sum(utilities_softmaxed))
	return utilities_softmaxed


def calculate_cost(utilities_softmaxed, n_samples, object_positions, intended_referent):
	cost_samples = np.zeros(n_samples)
	for i in range(n_samples):
		fixation_order = np.random.choice(object_positions, size=len(object_positions), replace=False, p=utilities_softmaxed)
		cost_samples[i] = fixation_order[intended_referent]
	negative_mean = -np.mean(cost_samples)
	return negative_mean



def speaker_production_probs(cost_per_word_per_referent, speaker_tau):
	prod_probs_softmaxed = np.exp(cost_per_word_per_referent/speaker_tau)
	prod_probs_softmaxed_normalized = np.divide(prod_probs_softmaxed, np.sum(prod_probs_softmaxed))
	return prod_probs_softmaxed_normalized






def pragmatic_listener(literal_speaker_production_probs_matrix, listener_tau):
	transposed_matrix = np.transpose(literal_speaker_production_probs_matrix)
	print("transposed_matrix is:")
	print(transposed_matrix)
	transposed_matrix_normalized = np.divide(transposed_matrix, np.sum(transposed_matrix))
	print("transposed_matrix_normalized is:")
	print(transposed_matrix_normalized)
	matrix_normalized = np.transpose(transposed_matrix_normalized)
	print("matrix_normalized is:")
	print(matrix_normalized)
	matrix_softmaxed = np.exp(matrix_normalized / listener_tau)
	print("matrix_softmaxed is:")
	print(matrix_softmaxed)
	matrix_softmaxed_normalized = np.divide(matrix_normalized, np.sum(matrix_normalized))
	print("matrix_softmaxed_normalized is:")
	print(matrix_softmaxed_normalized)
	return matrix_softmaxed_normalized


LS = LiteralSpeaker_MW.LiteralSpeaker(n_objects=len(object_positions), stau=speaker_rationality, ltau=listener_rationality, verbose=True)

for model in models:
	print('')
	print('')
	print('')
	print("model is:")
	print(model)
	for lpos in listener_positions:
		print("lpos is:")
		print(lpos)
		for words in [2, 3]:
			print('')
			print("words is:")
			print(words)

			literal_speaker_production_probs_matrix = np.zeros((len(object_positions), words))

			if words == 2:
				demonstrative_types = ["proximal", "distal"]
			elif words == 3:
				demonstrative_types = ["proximal", "medial" ,"distal"]

			for i in range(len(object_positions)):
				referent = object_positions[i]
				print('')
				print('')
				print("referent is:")
				print(referent)
				LS.SetEvent(method=model, referent=referent, lpos=lpos)

				utilities_literal_speaker = LS.ComputeUtilities(words)

				utilities_literal_speaker_my_calculation_julians_input = []
				utilities_literal_speaker_my_calculation_my_input = []
				for term in demonstrative_types:
					print('')
					print("term is:")
					print(term)
					utilities_per_referent = utilities_across_referents(model, LS.spos, lpos, term, object_positions)
					# print("utilities_per_referent is:")
					# print(utilities_per_referent)

					utilities_scaled_my_calculation = scale_utilities(utilities_per_referent)
					# print("utilities_scaled_my_calculation are:")
					# print(utilities_scaled_my_calculation)

					softmaxed_utilities_my_calculation_my_input = fixation_probs(utilities_scaled_my_calculation, listener_rationality)
					print("softmaxed_utilities_my_calculation_my_input are:")
					print(softmaxed_utilities_my_calculation_my_input)
					print("np.sum(softmaxed_utilities_my_calculation_my_input) are:")
					print(np.sum(softmaxed_utilities_my_calculation_my_input))

					cost_value_my_calculation_my_input = calculate_cost(softmaxed_utilities_my_calculation_my_input, 10000, object_positions, referent)

					utilities_literal_speaker_my_calculation_my_input.append(cost_value_my_calculation_my_input)

				print('')
				print("utilities_literal_speaker_my_calculation_my_input MY CALCULATION are:")
				print(utilities_literal_speaker_my_calculation_my_input)

				prod_probs_normalized_search_costs = normalize_search_costs(utilities_literal_speaker_my_calculation_my_input)
				print('')
				print("prod_probs_normalized_search_costs are:")
				print(prod_probs_normalized_search_costs)


				simple_speaker_prod_probs = speaker_production_probs(prod_probs_normalized_search_costs, speaker_rationality)
				print('')
				print("simple_speaker_prod_probs are:")
				print(simple_speaker_prod_probs)


				literal_speaker_production_probs_matrix[i] = simple_speaker_prod_probs


				print('')
				print("utilities_literal_speaker ORIGINAL CODE are:")
				print(utilities_literal_speaker)

				utilities_literal_speaker_my_calculation_normalized_my_input = np.divide(utilities_literal_speaker_my_calculation_my_input, np.sum(utilities_literal_speaker_my_calculation_my_input))
				print('')
				print("utilities_literal_speaker_my_calculation_normalized_my_input MY CALCULATION are:")
				print(utilities_literal_speaker_my_calculation_normalized_my_input)
				print("np.sum(utilities_literal_speaker_my_calculation_normalized_my_input) MY CALCULATION are:")
				print(np.sum(utilities_literal_speaker_my_calculation_normalized_my_input))


				utilities_literal_speaker_as_array = np.zeros(len(utilities_literal_speaker))
				for i in range(len(utilities_literal_speaker)):
					utilities_literal_speaker_as_array[i] = utilities_literal_speaker[i][-1]
				utilities_literal_speaker_normalized = np.divide(utilities_literal_speaker_as_array, np.sum(utilities_literal_speaker_as_array))
				print('')
				print("utilities_literal_speaker_normalized ORIGINAL CODE are:")
				print(utilities_literal_speaker_normalized)
				print("np.sum(utilities_literal_speaker_normalized) ORIGINAL CODE are:")
				print(np.sum(utilities_literal_speaker_normalized))


			PL = PragmaticListener.PragmaticListener(LS, words=words)

			PS = PragmaticSpeaker_MW.PragmaticSpeaker(PL, referent, output_dict)

			print('')
			print('')
			print('NOW THE EVENT IS RUN WITH THE PRAGMATIC SPEAKER:')
			PS.RunEvent()

		print('')
		print('')
		print("literal_speaker_production_probs_matrix is:")
		print(literal_speaker_production_probs_matrix)


		pragmatic_listener_matrix = pragmatic_listener(literal_speaker_production_probs_matrix, listener_rationality)
		print('')
		print('')
		print("pragmatic_listener_matrix is:")
		print(pragmatic_listener_matrix)
