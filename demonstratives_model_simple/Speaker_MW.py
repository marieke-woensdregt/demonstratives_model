import numpy as np
import sys

class Speaker:

	def __init__(self, method=None, referent=0, lpos=0, latt=0, stau=0.01, ltau=0.01, verbose=False):
		# Static prameters
		self.ObjectNo = 4 # Four objects!
		self.spos = 0 # speaker position
		# Dynamic parameters
		self.method = method
		self.referent = referent # an integer 0-3
		self.lpos = lpos # listener position
		self.latt = latt # listener attention
		self.stau = stau # speaker rationality
		self.ltau = ltau # listener rationality
		self.verbose = verbose
		# Semantic models
		self.modelcosts = {
		'distance':{'este':self.Este_distance,'ese':self.Ese_distance,'aquel':self.Aquel_distance},
		'person':{'este':self.Este_distance,'ese':self.Ese_person,'aquel':self.Aquel_person},
		'pdhybrid':{'este':self.Este_distance,'ese':self.Ese_distance,'aquel':self.Aquel_person},
		'person_attention':{'este':self.Este_attention,'ese':self.Ese_person,'aquel':self.Aquel_attention},
		'distance_attention':{'este':self.Este_attention,'ese':self.Ese_distance,'aquel':self.Aquel_attention},
		}

	def SetEvent(self, method=None, referent=0, lpos=0, latt=0):
		"""
		Set variables for the event
		"""
		if method is not None:
			self.method = method
		self.referent = referent
		self.lpos = lpos
		self.latt = latt

	def GetCost(self, distribution, samples=500):
		"""
		General function that gets costs given a probability distribution
		"""
		CostSamples = [0] * samples
		for i in range(samples):
			FixationOrder = np.random.choice(range(self.ObjectNo),size=self.ObjectNo,replace=False,p=distribution)
			# This produces a fixation order, so now the index of the right object corresponds to the cost.
			CostSamples[i] = list(FixationOrder).index(self.referent)
		return(-np.mean(CostSamples))

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
		methods = 'utilities' or 'visual search'.
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

	def Este_distance(self):
		"""
		# Look closest to the speaker
		"""
		# Use negative so that farther-away objects have lower utilities
		Distance=[-np.abs(x-self.spos) for x in range(self.ObjectNo)]
		Softmaxed = self.Softmax_Utilities(Distance)
		Costs = self.GetCost(Softmaxed)
		return(Costs)
		#return(self.GetCost(Softmaxed))

	def Ese_distance(self):
		"""
		# Look average distance from speaker
		"""
		# Prioritize average distance
		Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		Av = np.mean(Distance)
		# Now score each one based on how far it is from the average.
		# Use negative so that farther from average has lower utility
		DFromAv = [-np.abs(x-Av) for x in Distance]
		Softmaxed = self.Softmax_Utilities(DFromAv)
		return(self.GetCost(Softmaxed))

	def Aquel_distance(self):
		"""
		# Look farthest from speaker (inverse of Este-distance)
		"""
		# Use positive so that farther-away objects have higher utilities
		Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		Softmaxed = self.Softmax_Utilities(Distance)
		Costs = self.GetCost(Softmaxed)
		return(Costs)
		#return(self.GetCost(Softmaxed))

	def Ese_person(self):
		"""
		# Far from me and close to you
		"""
		# Speaker distance
		Speaker_Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		# Listener distance: negative so that farther-away objects have lower utilities
		Listener_Distance=[-np.abs(x-self.lpos) for x in range(self.ObjectNo)]
		Distance = [Speaker_Distance[x]+Listener_Distance[x] for x in range(len(Speaker_Distance))]
		# Now sample objects based on distance
		Softmaxed = self.Softmax_Utilities(Distance)
		return(self.GetCost(Softmaxed))

	def Aquel_person(self):
		"""
		# Far from both of us
		"""
		# Speaker distance
		Speaker_Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		# Listener distance
		Listener_Distance=[np.abs(x-self.lpos) for x in range(self.ObjectNo)]
		#print(str(Listener_Distance))
		Distance = [Speaker_Distance[x]+Listener_Distance[x] for x in range(len(Speaker_Distance))]
		Softmaxed = self.Softmax_Utilities(Distance)
		return(self.GetCost(Softmaxed))

	def Este_attention(self):
		"""
		# Este person with a boosted utility for pulling listener attention towards speaker
		"""
		if self.verbose:
			sys.stdout.write("ESTE\n")
		# PART 1: BASIC UTILITY
		# Use negative so that farther-away objects have lower utilities
		Distance=[-np.abs(x-self.spos) for x in range(self.ObjectNo)]
		# Now scale over 0-1 range and use these are core!
		Utilities_core = self.normalizeUtilities(Distance)
		if self.verbose:
			sys.stdout.write("\tmain utilities: "+str(np.round(Utilities_core,2))+"\n")
		# PART 2: ATTENTION UTILITY
		# speaker is looking farther away from object, so boost utilities
		Utilities_attention = [0]*self.ObjectNo
		if self.latt >= self.referent:
			Utilities_attention = [1 if x<self.latt else 0 for x in range(self.ObjectNo)]
			Utilities_attention = [x - min(Utilities_attention) for x in Utilities_attention] # will always be 0, but just to keep code consistent
		Utilities_attention = self.normalizeUtilities(Utilities_attention)
		if self.verbose:
			sys.stdout.write("\tattention-based utilites: "+str(np.round(Utilities_attention,2))+"\n")
		# PART 3: MERGE AND ESTIMATE COSTS
		Utilities = [sum(x) for x in zip(Utilities_core, Utilities_attention)]
		# Now sample objects based on distance!
		Softmaxed = self.Softmax_Utilities(Utilities)
		if self.verbose:
			sys.stdout.write("\tprobabilities of visual search: "+str(np.round(Softmaxed,2))+"\n")
		Costs = self.GetCost(Softmaxed)
		if self.verbose:
			sys.stdout.write("\t\tExpected search cost: "+str(np.round(Costs,2))+"\n")
		return(Costs)

	def Aquel_attention(self):
		"""
		# Compute utility of listener. "Aquel" means look further away from me!
		"""
		if self.verbose:
			sys.stdout.write("AQUEL\n")
		# PART 1: GET DISTANCE-BASED SCORE and scale
		Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		Utilities_core = self.normalizeUtilities(Distance)
		if self.verbose:
			sys.stdout.write("\tmain utilities: "+str(np.round(Utilities_core,2))+"\n")
		# PART 2: GET ATTENTION-BASED SCORE and scale
		# Speaker distance
		Speaker_Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		#print(str(Speaker_Distance))
		# Now get difference from attentional point
		Attention_Distance=[x-self.latt for x in Speaker_Distance]
		# Now just mark in which direction they go
		for x in range(len(Attention_Distance)):
			if Attention_Distance[x] > 0:
				Attention_Distance[x] = 1
			if Attention_Distance[x] < 0:
				Attention_Distance[x] = 0
				#Attention_Distance[x] = -1
		Utilities_attention = self.normalizeUtilities(Attention_Distance)
		if self.verbose:
			sys.stdout.write("\tattention-based utilites: "+str(np.round(Utilities_attention,2))+"\n")
		#print(str(Attention_Distance))
		# PART 3 COMBINE!
		Utilities = [sum(x) for x in zip(Utilities_core, Utilities_attention)]
		# Now sample objects based on distance!
		Softmaxed = self.Softmax_Utilities(Utilities)
		if self.verbose:
			sys.stdout.write("\tprobabilities of visual search: "+str(np.round(Softmaxed,2))+"\n")
		Costs = self.GetCost(Softmaxed)
		if self.verbose:
			sys.stdout.write("\t\tExpected search cost: "+str(np.round(Costs,2))+"\n")
		return(Costs)

	def ComputeUtilities(self, words):
		# Compute utilities associated with "este", "ese", and "aquel".
		if words==2:
			return([['este',self.modelcosts[self.method]['este']()],['aquel',self.modelcosts[self.method]['aquel']()]])
		else:
			return([['este',self.modelcosts[self.method]['este']()],['ese',self.modelcosts[self.method]['ese']()],['aquel',self.modelcosts[self.method]['aquel']()]])

	def RunEvent(self,referent=None,method=None,words=3,header=True):
		"""
		Update all items and compute utilities.
		words = 2 uses a two-word system (ese, aquel)
		attentionweight puts how much weight goes on the attention-correction mechanism. 1 means equal, 2 means twice as much!
		only works for attention model.
		"""
		if header:
			self.PrintHeader
		if referent is not None:
			self.referent = referent
		if method is not None:
			self.method = method
		if self.verbose:
			sys.stdout.write("\nS("+str(self.spos+1)+") talking to L("+str(self.lpos+1)+"/att="+str(self.latt+1)+") about "+str(self.referent+1)+"\n")
		values = self.ComputeUtilities(words)
		costs = [x[1] for x in values]
		softmaxed = self.Softmax_Utilities(costs,method='visualsearch',normalize=True)
		Details = str(self.referent)+","+str(self.spos)+","+str(self.lpos)+","+str(self.latt)+","+str(words)+","+str(self.stau)+","+str(self.ltau)
		if not self.verbose:
			sys.stdout.write(self.method+","+values[0][0]+","+str(np.round(softmaxed[0],2))+","+Details+"\n")
			sys.stdout.write(self.method+","+values[1][0]+","+str(np.round(softmaxed[1],2))+","+Details+"\n")
			if words==3:
				sys.stdout.write(self.method+","+values[2][0]+","+str(np.round(softmaxed[2],2))+","+Details+"\n")
		else:
			sys.stdout.write(values[0][0]+"("+str(np.round(softmaxed[0],2))+"), "+values[1][0]+"("+str(np.round(softmaxed[1],2))+")")
			if words==3:
				sys.stdout.write(", "+values[2][0]+"("+str(np.round(softmaxed[2],2))+")")
			sys.stdout.write("\n")

	def PrintHeader(self):
		sys.stdout.write("Model,Word,Probability,Referent,Speaker_pos,Listener_pos,Listener_att,WordNo,SpeakerTau,ListenerTau\n")
