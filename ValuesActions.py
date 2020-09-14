import numpy as np


class PolicyEvaluation:
	
	def __init__ (self):
		self.Vk = []
	
	def policy_evaluation (self, agent):
		"""
		:object agent: agent object
		:return:
		"""
		self.Vk = np.zeros(len(agent.states))
		while True:
			delta = 0
			for s, state in enumerate(agent.states):
				v = self.Vk[s]
				acc = 0
				for a, action in enumerate(agent.actions):
					for s1, _ in enumerate(agent.states):
						acc += agent.policy[s, a, :] * (
									agent.returns[s, a, :] + agent.gamma * self.Vk[s1])  # * prob(s',r| s,a)
				
				self.Vk[s] = acc
				delta = max(delta, abs(v - self.Vk[s]))
				if delta < agent.theta:
					break
