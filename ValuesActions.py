class ValuesActions:
	
	def __init__ (self):
		self.Vk = []
	
	def policy_evaluation (self, agent):
		"""
		:object agent: agent object
		:return:
		"""
		self.Vk = agent.state_values
		while True:
			for s, state in enumerate(agent.states):
				delta = 0
				if s in set(agent.terminal_states_idxs):
					pass
				else:
					v = self.Vk[s]
					acc = 0
					for a, action in enumerate(agent.actions):
						for s1, _ in enumerate(agent.states):
							for r, r_value in enumerate(agent.returns):
								acc += agent.policy[s, a] * \
									agent.s1_r_probs[s, a, s1, r] * \
									(r_value + agent.gamma * self.Vk[s1])
					
					self.Vk[s] = acc
					delta = max(delta, abs(v - self.Vk[s]))
					if delta < agent.theta:
						return self.Vk
