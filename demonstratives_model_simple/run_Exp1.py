import Speaker

listener_rationality = 1
speaker_rationality = 1

Model = Speaker.Speaker(stau=speaker_rationality,ltau=listener_rationality,verbose=False)

Model.PrintHeader()
# for method in ['distance','person'9,'pdhybrid']:
for method in ['person_attention','distance_attention']:
	for lpos in [0,1,2,3]:
		for referent in [0,1,2,3]:
			for words in [2,3]:
				Model.SetEvent(method=method, referent=referent, lpos=lpos)
				Model.RunEvent(words=words,header=False)

