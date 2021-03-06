import numpy as np
import sys

class PragmaticListener:

	def __init__(self, LS,words):
		# LS is a literalspeaker object
		self.LS = LS
		# after self.ComputeMatrix this is a dict for referents and then has 'este', 'ese' (if words=3), 'aquel'
		self.UtilityMatrix = {}
		self.words = words # 2- or 3-word system?
		self.searchstrategy = [0]*LS.ObjectNo
		# Initialize utility matrix:
		self.UtilityMatrix['este']=[0]*LS.ObjectNo
		if self.words==3:
			self.UtilityMatrix['ese']=[0]*LS.ObjectNo
		self.UtilityMatrix['aquel']=[0]*LS.ObjectNo

	def ComputeMatrix(self):
		"""
		Compute a matrix beween utterances and referents
		"""

		# sys.stdout.write("\n")
		# sys.stdout.write("this is the ComputeMatrix method of the PragmaticListener class:")
		# sys.stdout.write("\n")

		for referent in range(self.LS.ObjectNo):
			self.LS.referent = referent

			# sys.stdout.write("\n")
			# sys.stdout.write("referent in PL.ComputeMatrix() is:")
			# sys.stdout.write("\n")
			# sys.stdout.write(str(referent))
			# sys.stdout.write("\n")

			values = self.LS.ComputeUtilities(self.words)

			# sys.stdout.write("\n")
			# sys.stdout.write("values in PL.ComputeMatrix() are; THESE SHOULD BE THE RAW COST VALUES GIVEN BY LS.ComputeUtilities():")
			# sys.stdout.write("\n")
			# sys.stdout.write(str(values))
			# sys.stdout.write("\n")

			costs = [x[1] for x in values]
			utterances = [x[0] for x in values]
			softmaxed = self.LS.Softmax_Utilities(costs,method='visualsearch')

			# sys.stdout.write("\n")
			# sys.stdout.write("softmaxed in PL.ComputeMatrix() is:")
			# sys.stdout.write("\n")
			# sys.stdout.write(str(softmaxed))
			# sys.stdout.write("\n")

			# TEMPO CODE FOR UNDERSTANDING SOFTMAX
			#temp = [np.round(x,2) for x in softmaxed]
			#sys.stdout.write(str(temp)+"\n")
			# ENDS ABOVE!
			for u_i in range(len(utterances)):
				self.UtilityMatrix[utterances[u_i]][referent] = softmaxed[u_i]

	def PragmaticVisualSearch(self, utterance):
		# If you said "utterance" which object is it worth looking at?

		# sys.stdout.write("\n")
		# sys.stdout.write("self.UtilityMatrix given as input to PL.PragmaticVisualSearch are:")
		# sys.stdout.write("\n")
		# sys.stdout.write(str(self.UtilityMatrix))
		# sys.stdout.write("\n")

		for referent in range(self.LS.ObjectNo):
			#let's suppose its referent, what's the chance you
			# woudl've chosen the utterance?
			self.searchstrategy[referent]=self.UtilityMatrix[utterance][referent]*1.0/sum([self.UtilityMatrix[x][referent] for x in self.UtilityMatrix.keys()])
		self.searchstrategy = [self.searchstrategy[x]/sum(self.searchstrategy) for x in range(len(self.searchstrategy))]


