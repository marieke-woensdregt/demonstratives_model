import numpy as np
import sys

class LiteralSpeaker:

	def __init__(self, n_objects=4, method=None, referent=0, lpos=0, latt=0, stau=0.01, ltau=0.01, wobj=0.1, wlist=0.1, verbose=False):
		# Static prameters
		self.ObjectNo = n_objects  # Number of objects
		self.spos = 0
		# Dynamic parameters
		self.method = method
		self.referent = referent
		self.lpos = lpos
		self.latt = latt
		self.stau = stau # speaker rationality
		self.ltau = ltau # listener rationality
		self.wobj = wobj # weight of object location parameter
		self.wlist = wlist # weight of listeren location parameter
		self.verbose = verbose
		# Semantic models
		self.modelcosts = {
		'distance':{'este':self.Este_distance,'ese':self.Ese_distance,'aquel':self.Aquel_distance},
		'person':{'este':self.Este_distance,'ese':self.Ese_person,'aquel':self.Aquel_person},
		'pdhybrid':{'este':self.Este_distance,'ese':self.Ese_distance,'aquel':self.Aquel_person},
		'person_attention':{'este':self.Este_attention,'ese':self.Ese_person,'aquel':self.Aquel_person_attention},  #MW: Split the Aquel_attention() method into two separate methods: Aquel_person_attention() and Aquel_distance_attention
		'distance_attention':{'este':self.Este_attention,'ese':self.Ese_distance,'aquel':self.Aquel_distance_attention},
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

	def GetCost(self, distribution, samples=10000): #TODO: CHANGE THIS BACK TO 1000
		"""
		General function that gets costs given a probability distribution
		"""

		# sys.stdout.write("\n")
		# sys.stdout.write("distribution given as input to LS.GetCost() is:")
		# sys.stdout.write("\n")
		# sys.stdout.write(str(distribution))
		# sys.stdout.write("\n")
		# sys.stdout.write("str(np.sum(distribution)) is:")
		# sys.stdout.write("\n")
		# sys.stdout.write(str(np.sum(distribution)))
		# sys.stdout.write("\n")

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

		# sys.stdout.write("\n")
		# sys.stdout.write("\n")
		# sys.stdout.write("THIS IS WHEN THE normalizeSearchCost() METHOD GETS CALLED!:")
		# sys.stdout.write("\n")
		# sys.stdout.write("\n")

		utilities_scaled = [x+3 for x in values] # shift by 3 so we're on a 0-3 scale
		return(utilities_scaled)

	def Softmax_Utilities(self, utilities, method='utilities', normalize=True):
		"""
		General function to softmax a utility function.
		When normalize is true, utilities are first normalized.
		methods = 'utilities' or 'visual search'.
		For utilities it means you're getting word valuers. visual search means you're getting visual cost estimates
		"""

		# sys.stdout.write("\n")
		# sys.stdout.write("These are the utilities given as input to LS.Softmax_Utilities() BEFORE NORMALIZING:")
		# sys.stdout.write("\n")
		# sys.stdout.write(str(utilities))
		# sys.stdout.write("\n")


		if method=='utilities':
			tau = self.ltau
		else:
			tau = self.stau #speaker
		if normalize:
			if method=='utilities':
				utilities = self.normalizeUtilities(utilities)
			else:
				utilities = self.normalizeSearchCost(utilities)

		#
		# sys.stdout.write("\n")
		# sys.stdout.write("These are the utilities given as input to LS.Softmax_Utilities() AFTER NORMALIZING:")
		# sys.stdout.write("\n")
		# sys.stdout.write(str(utilities))
		# sys.stdout.write("\n")


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
		# Now sample objects based on distance
		Softmaxed = self.Softmax_Utilities(Distance)

		# sys.stdout.write("\n")
		# sys.stdout.write("term is proximal")
		# sys.stdout.write("\n")

		return(self.GetCost(Softmaxed))

	def Ese_distance_OLD(self):
		"""
		# Look average distance from speaker
		"""
		# Prioritize average distance
		Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		Av = np.mean(Distance)
		# Now score each one based on how far it is from the average.
		# Use negative so that farther from average has lower utility
		DFromAv = [-np.abs(x-Av) for x in Distance]
		# Now sample objects based on this metric!
		Softmaxed = self.Softmax_Utilities(DFromAv)
		return(self.GetCost(Softmaxed))

	def Ese_distance(self):
		"""
		# New 'neutral' ese. Allow RSA to center its meaning
		"""
		# Prioritize average distance
		Utilities = [1] * self.ObjectNo
		# Now sample objects based on this metric!
		Softmaxed = self.Softmax_Utilities(Utilities)

		# sys.stdout.write("\n")
		# sys.stdout.write("term is medial")
		# sys.stdout.write("\n")

		return(self.GetCost(Softmaxed))

	def Aquel_distance(self,):
		"""
		# Look farthest from speaker (inverse of Este-distance)
		"""
		# Use positive so that farther-away objects have higher utilities
		Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		# Now sample objects based on distance
		Softmaxed = self.Softmax_Utilities(Distance)

		# sys.stdout.write("\n")
		# sys.stdout.write("term is distal")
		# sys.stdout.write("\n")

		return(self.GetCost(Softmaxed))


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

		# sys.stdout.write("\n")
		# sys.stdout.write("term is medial")
		# sys.stdout.write("\n")

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
		# Now sample objects based on distance!
		Softmaxed = self.Softmax_Utilities(Distance)

		# sys.stdout.write("\n")
		# sys.stdout.write("term is distal")
		# sys.stdout.write("\n")

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

		# sys.stdout.write("\n")
		# sys.stdout.write("term is proximal")
		# sys.stdout.write("\n")

		if self.verbose:
			sys.stdout.write("\tprobabilities of visual search: "+str(np.round(Softmaxed,2))+"\n")
		Costs = self.GetCost(Softmaxed)
		if self.verbose:
			sys.stdout.write("\t\tExpected search cost: "+str(np.round(Costs,2))+"\n")
		return(Costs)

	def Aquel_distance_attention(self):
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

		# sys.stdout.write("\n")
		# sys.stdout.write("term is distal")
		# sys.stdout.write("\n")

		if self.verbose:
			sys.stdout.write("\tprobabilities of visual search: "+str(np.round(Softmaxed,2))+"\n")
		Costs = self.GetCost(Softmaxed)
		if self.verbose:
			sys.stdout.write("\t\tExpected search cost: "+str(np.round(Costs,2))+"\n")
		return(Costs)

	def Aquel_person_attention(self):
		"""
		# Compute utility of listener. "Aquel" means look further away from me!
		"""
		if self.verbose:
			sys.stdout.write("AQUEL\n")
		# PART 1: GET PERSON-BASED SCORE and scale

		##############################################################################################
		# BIT THAT MARIEKE CHANGED:
		# Just replaced the part that was taken from the Aquel_distance() method and insterted in the
		# Aquel_distance_attention() method above, and instead inserted the equivalent part of the
		# Aquel_person() method below. Left the rest the same.

		# Speaker distance
		Speaker_Distance=[np.abs(x-self.spos) for x in range(self.ObjectNo)]
		# Listener distance
		Listener_Distance=[np.abs(x-self.lpos) for x in range(self.ObjectNo)]
		#print(str(Listener_Distance))
		Distance = [Speaker_Distance[x]+Listener_Distance[x] for x in range(len(Speaker_Distance))]
		Utilities_core = self.normalizeUtilities(Distance)
		if self.verbose:
			sys.stdout.write("\tmain utilities: "+str(np.round(Utilities_core,2))+"\n")

		##############################################################################################

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

		# sys.stdout.write("\n")
		# sys.stdout.write("term is distal")
		# sys.stdout.write("\n")

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


