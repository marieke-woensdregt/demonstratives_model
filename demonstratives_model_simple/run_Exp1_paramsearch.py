import Speaker
import numpy as np

listener_rationality = 1
speaker_rationality = 2

Model = Speaker.Speaker(stau=speaker_rationality,ltau=listener_rationality,verbose=False)

Model.PrintHeader()

for listener_rationality in np.arange(0.2,0.4,0.01):
	for speaker_rationality in np.arange(0.2,0.4,0.01):
		Model = Speaker.Speaker(stau=speaker_rationality,ltau=listener_rationality,verbose=False)
		for method in ['distance','person','pdhybrid']:
			for lpos in [0,1,2,3]:
				for referent in [0,1,2,3]:
					for words in [2,3]:
						Model.SetEvent(method=method, referent=referent, lpos=lpos)
						Model.RunEvent(words=words,header=False)
