import LiteralSpeaker_weighted_MW
import PragmaticListener
import PragmaticSpeaker_weighted_MW
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

tau_start = 0.4
tau_stop = 2.05
tau_step = 0.05

weight_obj_start = 0.0
weight_obj_stop = 1.0
weight_obj_step = 0.1

listener_positions = [0,1,2,3,4]
object_positions = [0,1,2,3,4]

models = ['distance', 'person']  # Can contain: 'distance','person','pdhybrid', 'distance_attention', 'person_attention'

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
			   "ListenerTau":[],
			   "WeightObject": [],
			   "WeightListener": []
			   }

########################################################################################

# sys.stdout.write("Model,Word,Probability,Referent,Speaker_pos,Listener_pos,Listener_att,WordNo,SpeakerTau,ListenerTau\n")

# for listener_rationality in itertools.chain(np.arange(0.01,0.1,0.01),np.arange(0.1,1.2,0.2)):
for listener_rationality in np.arange(tau_start, tau_stop, tau_step):
	print('')
	print(f"listener_rationality is {listener_rationality}:")
	# for speaker_rationality in np.arange(0.1,1,0.1):
	for speaker_rationality in np.arange(tau_start, tau_stop, tau_step):
		print(f"speaker_rationality is {speaker_rationality}:")
		for object_weight in np.arange(weight_obj_start, weight_obj_stop, weight_obj_step):
			listener_weight = 1.0-object_weight
			print(f"object_weight is {object_weight}:")
			print(f"listener_weight is {listener_weight}:")
			LS = LiteralSpeaker_weighted_MW.LiteralSpeaker(n_objects=len(object_positions), stau=speaker_rationality,ltau=listener_rationality, wobj=object_weight, wlist=listener_weight, verbose=False) #TODO: Move the rounding to here instead of elsewhere?
			for model in models:
				for lpos in listener_positions:
					for referent in object_positions:
						LS.SetEvent(method=model, referent=referent, lpos=lpos)
						for words in [2,3]:
							PL = PragmaticListener.PragmaticListener(LS,words=words)
							PS = PragmaticSpeaker_weighted_MW.PragmaticSpeaker(PL,referent, output_dict)
							PS.RunEvent()
			output_dict = PS.output_dict  # MW: has to be updated every time, so each new speaker gets updated with the existing output_dict, and new data is written to the existing output_dict


output_dataframe = pd.DataFrame(data=output_dict)
pd.set_option('display.max_columns', None)
# print('')
# print('')
# print(output_dataframe)
# print('')
# print(output_dataframe.columns)


# output_file_path = '/Users/U968195/PycharmProjects/demonstratives_model/model_predictions/'
output_file_path = 'model_predictions/'
output_file_name = 'HigherSearchD_MW_RSA_n_positions_'+str(len(object_positions)) + str(models).replace(" ", "") + '_tau_start_' + str(tau_start) + '_tau_stop_' + str(tau_stop) + '_tau_step_' + str(tau_step) + '_wobj_start_' + str(weight_obj_start) + '_wobj_stop_' + str(weight_obj_stop) + '_wobj_step_' + str(weight_obj_step) +'.csv'
output_dataframe.to_csv(output_file_path+output_file_name, index=False)

# end time:
end = time.time()

# total time taken:
print(f"Runtime of the program is {round(((end - start)/60), 2)} minutes")
