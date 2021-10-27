import LiteralSpeaker
import PragmaticListener
import PragmaticSpeaker
import sys
import numpy as np
import itertools

itertools.chain(np.arange(0.1,0.5,0.1),np.arange(0.1,0.5,0.1))

# LEARNED FROM SIMULATIONS:

# FOR LISTENER: 0.1 PUTS .96 ON CLOSEST FOR 'ESTE'
# FOR LISTENER: 1.1 PUTS .37 ON CLOSEST FOR 'ESTE'

# FOR SPEAKER: >1 IT'S JUST CHANCE. RANGE SEEMS TO BE 0.1 TO 0.5

sys.stdout.write("Model,Word,Probability,Referent,Speaker_pos,Listener_pos,Listener_att,WordNo,SpeakerTau,ListenerTau\n")

for listener_rationality in itertools.chain(np.arange(0.01,0.1,0.01),np.arange(0.1,1.2,0.2)):
	for speaker_rationality in np.arange(0.1,1,0.1):
		LS = LiteralSpeaker.LiteralSpeaker(stau=speaker_rationality,ltau=listener_rationality,verbose=False)
		for method in ['distance','person','pdhybrid']:
			for lpos in [0,1,2,3]:
				for referent in [0,1,2,3]:
					LS.SetEvent(method=method, referent=referent, lpos=lpos)
					for words in [2,3]:
						PL = PragmaticListener.PragmaticListener(LS,words=words)
						PS = PragmaticSpeaker.PragmaticSpeaker(PL,referent)
						PS.RunEvent()
