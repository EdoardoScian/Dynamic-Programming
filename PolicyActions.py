import numpy as np


class PolicyActions:
	
	def __init__ (self):
		self.action_values = []
	
	def policy_improvement (self, agent):
		"""
		:object agent: agent object
		:return:
		"""
		stable_policy = True
		self.action_values = agent.action_values
		for s, state in enumerate(agent.states):
			if s in set(agent.terminal_states_idxs):
				pass
			else:
				old_actions = np.argwhere(agent.policy[s, :] == np.amax(agent.policy[s, :]))[:, 0]
				for a, action in enumerate(agent.actions):
					acc = 0
					for s1, _ in enumerate(agent.states):
						for r, r_value in enumerate(agent.returns):
							acc += agent.s1_r_probs[s, a, s1, r] * (r_value + agent.gamma * agent.state_values[s1])
						if acc != 0 or (acc == 0 and s1 in set(agent.terminal_states_idxs) and
						                agent.s1_r_probs[s, a, s1, 0] != 0):
							self.action_values[s, a] = acc
				
				rounded_values = np.around(self.action_values[s, :], 2)
				new_actions = np.argwhere(rounded_values == np.amax(rounded_values))[:, 0]
				if set(new_actions) != set(old_actions):
					if len(new_actions) > 1:
						perc = 1 / len(new_actions)
						agent.policy[s, :] = 0
						for action in new_actions:
							agent.policy[s, action] = perc
					else:
						agent.policy[s, :] = 0
						agent.policy[s, new_actions] = 1
						stable_policy = False
		
		if stable_policy:
			agent.train = False
		
		return self.action_values
