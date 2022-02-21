import Speaker_MW
import numpy as np
import pandas as pd
import time

# starting time:
start = time.time()

listener_rationality = 1.
speaker_rationality = 2.

########################################################################################
# MW added code below:

tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

models = ['distance_attention', 'person_attention']  # ['distance', 'person'] # ['distance_attention', 'person_attention']  # Can contain: 'distance','person','pdhybrid', 'distance_attention', 'person_attention'

ese_uniform = True  # Can be set to either True or False. Determines whether "ese" under the simple distance model is a uniform distribution (if set to True), or rather centred around the medial objects (if set to False)

output_dict = {"Model":[],
			   "Word":[],
			   "Cost":[],
			   "Probability":[],
			   "Referent":[],
			   "Speaker_pos":[],
			   "Listener_pos":[],
			   "Listener_att":[],
			   "WordNo":[],
			   "SpeakerTau":[],
			   "ListenerTau":[]}
########################################################################################

Model = Speaker_MW.Speaker(output_dict, stau=speaker_rationality, ltau=listener_rationality, verbose=False, uniform_ese=ese_uniform)

# Model.PrintHeader()  # MW: Got rid of this, because everything is being saved to a pandas dataframe now.

for listener_rationality in np.arange(tau_start, tau_stop, tau_step):
	# print('')
	# print(f"listener_rationality is {listener_rationality}:")
	for speaker_rationality in np.arange(tau_start, tau_stop, tau_step):
		# print(f"speaker_rationality is {speaker_rationality}:")
		Model = Speaker_MW.Speaker(output_dict, stau=speaker_rationality,ltau=listener_rationality,verbose=False, uniform_ese=ese_uniform)
		for model in models:
			for latt in [0,1,2,3]:
				for referent in [1,2,3]:  # If I understood the design correctly, Exp. 2 only uses object positions [1, 2, 3]
					for words in [2,3]:
						Model.SetEvent(method=model, referent=referent, lpos=referent, latt=latt)
						Model.RunEvent(words=words,header=False)
		output_dict = Model.output_dict # MW: has to be updated every time, so each new speaker gets updated with the existing output_dict, and new data is written to the existing output_dict


output_dataframe = pd.DataFrame(data=output_dict)
pd.set_option('display.max_columns', None)
# print('')
# print('')
# print(output_dataframe)
# print('')
# print(output_dataframe.columns)


# output_file_path = '/Users/U968195/PycharmProjects/demonstratives_model/model_predictions/'
output_file_path = 'model_predictions/'
output_file_name = 'HigherSearchD_MW_Simple_Attention_Ese_uniform_'+ str(ese_uniform) + '_' + str(models).replace(" ", "") + '_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv'
output_dataframe.to_csv(output_file_path+output_file_name, index=False)


# end time:
end = time.time()

# total time taken:
print(f"Runtime of the program is {round(((end - start)/60), 2)} minutes")