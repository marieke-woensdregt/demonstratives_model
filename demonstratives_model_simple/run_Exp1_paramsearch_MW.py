import Speaker_MW
import numpy as np
import pandas as pd
import time

# starting time:
start = time.time()

listener_rationality = 1.
speaker_rationality = 2.

tau_start = 0.1
tau_stop = 0.61
tau_step = 0.02

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

Model = Speaker_MW.Speaker(output_dict, stau=speaker_rationality, ltau=listener_rationality, verbose=False)

# Model.PrintHeader()  # MW: Got rid of this, because everything is being saved to a pandas dataframe now.

for listener_rationality in np.arange(tau_start, tau_stop, tau_step):
	print('')
	print(f"listener_rationality is {listener_rationality}:")
	for speaker_rationality in np.arange(tau_start, tau_stop, tau_step):
		print(f"speaker_rationality is {speaker_rationality}:")
		Model = Speaker_MW.Speaker(output_dict, stau=speaker_rationality,ltau=listener_rationality,verbose=False)
		for method in ['distance','person','pdhybrid']:
			for lpos in [0,1,2,3]:
				for referent in [0,1,2,3]:
					for words in [2,3]:
						Model.SetEvent(method=method, referent=referent, lpos=lpos)
						Model.RunEvent(words=words,header=False)
		output_dict = Model.output_dict # MW has to be updated every time, so each new speaker gets updated with the existing output_dict, and new data is written to the existing output_dict



output_dataframe = pd.DataFrame(data=output_dict)
pd.set_option('display.max_columns', None)
print('')
print('')
print(output_dataframe)
print('')
print(output_dataframe.columns)


output_file_path = '/Users/U968195/PycharmProjects/demonstratives_model/model_predictions/'
output_file_name = 'HigherSearchD_MW'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv'
output_dataframe.to_csv(output_file_path+output_file_name, index=False)

# end time:
end = time.time()

# total time taken:
print(f"Runtime of the program is {round(((end - start)/60), 2)} minutes")