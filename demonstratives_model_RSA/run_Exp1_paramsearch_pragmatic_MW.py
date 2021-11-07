import LiteralSpeaker
import PragmaticListener
import PragmaticSpeaker_MW
import sys
import numpy as np
import itertools
import time
import pandas as pd


########################################################################################
# JJE's comments:

# itertools.chain(np.arange(0.1,0.5,0.1),np.arange(0.1,0.5,0.1))

# LEARNED FROM SIMULATIONS:

# FOR LISTENER: 0.1 PUTS .96 ON CLOSEST FOR 'ESTE'
# FOR LISTENER: 1.1 PUTS .37 ON CLOSEST FOR 'ESTE'

# FOR SPEAKER: >1 IT'S JUST CHANCE. RANGE SEEMS TO BE 0.1 TO 0.5
########################################################################################


########################################################################################
# MW added code below:

# starting time:
start = time.time()

tau_start = 0.1
tau_stop = 1.1
tau_step = 0.5

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

# sys.stdout.write("Model,Word,Probability,Referent,Speaker_pos,Listener_pos,Listener_att,WordNo,SpeakerTau,ListenerTau\n")

# for listener_rationality in itertools.chain(np.arange(0.01,0.1,0.01),np.arange(0.1,1.2,0.2)):
for listener_rationality in np.arange(tau_start, tau_stop, tau_step):
	print('')
	print(f"listener_rationality is {listener_rationality}:")
	# for speaker_rationality in np.arange(0.1,1,0.1):
	for speaker_rationality in np.arange(tau_start, tau_stop, tau_step):
		print(f"speaker_rationality is {speaker_rationality}:")
		LS = LiteralSpeaker.LiteralSpeaker(stau=speaker_rationality,ltau=listener_rationality,verbose=False) #TODO: Move the rounding to here instead of elsewhere?
		for method in ['distance','person','pdhybrid']:
			for lpos in [0,1,2,3]:
				for referent in [0,1,2,3]:
					LS.SetEvent(method=method, referent=referent, lpos=lpos)
					for words in [2,3]:
						PL = PragmaticListener.PragmaticListener(LS,words=words)
						PS = PragmaticSpeaker_MW.PragmaticSpeaker(PL,referent, output_dict)
						PS.RunEvent()
		output_dict = PS.output_dict  # MW: has to be updated every time, so each new speaker gets updated with the existing output_dict, and new data is written to the existing output_dict

print('')
print('')
print(output_dict)
for key, value in output_dict.items():
	print('')
	print("key is:")
	print(key)
	print("value is:")
	print(value)
	print("len(value) is:")
	print(len(value))

output_dataframe = pd.DataFrame(data=output_dict)
pd.set_option('display.max_columns', None)
print('')
print('')
print(output_dataframe)
print('')
print(output_dataframe.columns)


output_file_path = '/Users/U968195/PycharmProjects/demonstratives_model/model_predictions/'
output_file_name = 'HigherSearchD_MW_RSA_'+'_tau_start_'+str(tau_start)+'_tau_stop_'+str(tau_stop)+'_tau_step_'+str(tau_step)+'.csv'
output_dataframe.to_csv(output_file_path+output_file_name, index=False)

# end time:
end = time.time()

# total time taken:
print(f"Runtime of the program is {round(((end - start)/60), 2)} minutes")
