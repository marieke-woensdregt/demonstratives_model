import LiteralSpeaker
import PragmaticListener
import PragmaticSpeaker

listener_rationality = 1
speaker_rationality = 2

LS = LiteralSpeaker.LiteralSpeaker(stau=speaker_rationality,ltau=listener_rationality,verbose=False)
LS.SetEvent(method='distance', referent=0, lpos=2)
PL = PragmaticListener.PragmaticListener(LS,words=3)
PS = PragmaticSpeaker.PragmaticSpeaker(PL,0)
PS.SelectUtterance(0)

Model.PrintHeader()

for listener_rationality in np.arange(0.1,2,0.2):
	for speaker_rationality in np.arange(0.1,2,0.2):
		LS = LiteralSpeaker.LiteralSpeaker(stau=speaker_rationality,ltau=listener_rationality,verbose=False)
		for method in ['distance','person','pdhybrid']:
			for lpos in [0,1,2,3]:
				for referent in [0,1,2,3]:
					LS.SetEvent(method=method, referent=referent, lpos=lpos)
					for words in [2,3]:
						PL = PragmaticListener.PragmaticListener(LS,words=words)
						PS = PragmaticSpeaker.PragmaticSpeaker(PL,0)
						PS.SelectUtterance(0)
