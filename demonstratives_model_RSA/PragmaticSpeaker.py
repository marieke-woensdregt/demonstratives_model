import numpy as np
import sys
import PragmaticListener
import LiteralSpeaker

class PragmaticSpeaker:

	def __init__(self,PL, target):
		self.PL=PL
		self.stau = self.PL.LS.stau # extract speaker rationality
		self.words = self.PL.words
		self.ObjectNo = self.PL.LS.ObjectNo
		self.target = target

	def SelectUtterance(self,target=None):
		if target is not None:
			self.target=target
		values = self.ComputeUtilities()
		costs = [x[1] for x in values]
		softmaxed = self.Softmax_Utilities(costs, method='visualsearch', normalize=True)
		return(softmaxed)
		#sys.stdout.write(str(softmaxed))

	def ComputeUtilities(self):
		self.PL.ComputeMatrix()
		# Get cost for first word:
		self.PL.PragmaticVisualSearch('este')
		estecost = self.GetCost(self.PL.searchstrategy)
		a = ['este',estecost]
		if self.words==3:
			self.PL.PragmaticVisualSearch('ese')
			esecost = self.GetCost(self.PL.searchstrategy)
			b = ['ese',esecost]
		self.PL.PragmaticVisualSearch('aquel')
		aquelcost = self.GetCost(self.PL.searchstrategy)
		c = ['aquel',aquelcost]
		if self.words==3:
			return([a,b,c])
		else:
			return([a,c])

	def normalizeUtilities(self, values):
		"""
		Simple function to normalize a utility function
		"""
		utilities_scaled = [x - min(values) for x in values]
		if sum(utilities_scaled)==0:
			return [1.0/len(utilities_scaled)] * len(utilities_scaled)
		else:
			utilities_scaled = [x*1.0/max(utilities_scaled) for x in utilities_scaled]
		#sys.stdout.write(str(utilities_scaled)+"\n")
		return(utilities_scaled)

	def normalizeSearchCost(self, values):
		"""
		Simple function to normalize a utility function
		"""
		utilities_scaled = [x+3 for x in values] # shift by 3 so we're on a 0-3 scale
		return(utilities_scaled)

	def Softmax_Utilities(self, utilities, method='utilities', normalize=True):
		"""
		General function to softmax a utility function.
		When normalize is true, utilities are first normalized.
		methods = 'utilities' or 'visualsearch'.
		For utilities it means you're getting word valuers. visual search means you're getting visual cost estimates
		"""
		if method=='utilities':
			tau = self.ltau
		else:
			tau = self.stau #speaker
		if normalize:
			if method=='utilities':
				utilities = self.normalizeUtilities(utilities)
			else:
				utilities = self.normalizeSearchCost(utilities)
		if sum(utilities)==0:
			return [1.0/len(utilities)] * len(utilities)
		Softmaxed = [np.exp(x*1.0/tau) for x in utilities]
		Softmaxed = [x/sum(Softmaxed) for x in Softmaxed]
		return(Softmaxed)

	def PrintHeader(self):
		sys.stdout.write("Model,Word,Probability,Referent,Speaker_pos,Listener_pos,Listener_att,WordNo,SpeakerTau,ListenerTau\n")

	def GetCost(self, distribution, samples=100):
		"""
		General function that gets costs given a probability distribution
		"""
		CostSamples = [0] * samples
		for i in range(samples):
			FixationOrder = np.random.choice(range(self.ObjectNo),size=self.ObjectNo,replace=False,p=distribution)
			# This produces a fixation order, so now the index of the right object corresponds to the cost.
			CostSamples[i] = list(FixationOrder).index(self.target)
		return(-np.mean(CostSamples))

	def RunEvent(self):
		"""
		Update all items and compute utilities.
		words = 2 uses a two-word system (ese, aquel)
		attentionweight puts how much weight goes on the attention-correction mechanism. 1 means equal, 2 means twice as much!
		only works for attention model.
		"""	
		probabilities = self.SelectUtterance()
		if self.words==2:
			utterances = ['este','aquel']
		else:
			utterances = ['este','ese','aquel']
		Details = str(self.target)+","+str(self.PL.LS.spos)+","+str(self.PL.LS.lpos)+","+str(self.PL.LS.latt)+","+str(self.words)+","+str(self.stau)+","+str(self.PL.LS.ltau)
		sys.stdout.write(self.PL.LS.method+","+utterances[0]+","+str(np.round(probabilities[0],2))+","+Details+"\n")
		sys.stdout.write(self.PL.LS.method+","+utterances[1]+","+str(np.round(probabilities[1],2))+","+Details+"\n")
		if self.words==3:
			sys.stdout.write(self.PL.LS.method+","+utterances[2]+","+str(np.round(probabilities[2],2))+","+Details+"\n")

