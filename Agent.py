import numpy as np
import ValuesActions


class Agent:
	
	def __init__ (self, policy, states, actions, returns):
		"""
		:numpy policy: 3D cube (x,y,z) with (states, actions, probability)
		:list<string> states: MDP states list
		:list<string> actions: MDP actions list
		:numpy returns: 3D cube (x,y,z) with (states, actions, return)
		"""
		self.policy = policy
		self.states = states
		self.actions = actions
		self.returns = returns
		self.state_values = []
		self.action_values = []
		self.gamma = 1
		self.theta = 0.001
		self.valAct = ValuesActions.PolicyEvaluation()
	
	def policy_iteration (self, gamma, theta):
		"""
		:float gamma: 0 (myopic) <= gamma <= 1 (infinitely farsighted)
		:float theta: << 1 values update delta in order to stop value iteration
		:return:
		"""
		self.gamma = gamma
		self.theta = theta
		self.valAct.policy_evaluation(self)
