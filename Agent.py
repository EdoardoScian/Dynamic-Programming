import numpy as np
from ValuesActions import ValuesActions
from PolicyActions import PolicyActions


class Agent:
	
	def __init__ (self, s1_r_probs, policy, states_infos, actions, returns):
		"""
		:numpy s1_r_probs: 5D numpy of shape (x, y, z, v, 1) with (states_idx, actions_idx, states_idx, returns_idx, probability)
		:numpy policy: 3D numpy  of shape (x,y,1) with (states_idx, actions_idx, probability)
		:tuple(list<string>, list<int>) states_infos: list of states and list of terminal states indexes
		:list<string> actions: MDP actions list
		:list<float> returns: MDP returns list. 0-index element is the terminal states return
		"""
		self.policy = policy
		self.s1_r_probs = s1_r_probs
		self.states = states_infos[0]
		self.terminal_states_idxs = states_infos[1]
		self.actions = actions
		self.returns = returns
		self.state_values = np.zeros((len(states_infos[0])))
		self.action_values = np.zeros((len(states_infos[0]), len(actions), 1))
		self.action_values[:, :, :] = -9999999
		for i in self.terminal_states_idxs:
			self.action_values[i, :, :] = 0
		self.gamma = None
		self.theta = None
		self.valAct = ValuesActions()
		self.polAct = PolicyActions()
		self.train = True
	
	def get_actual_policy (self):
		for s, state in enumerate(self.states):
			print('Policy for state {}:\n'.format(state) + '{')
			for a, action in enumerate(self.actions):
				print('    Action: {} --> Probability: {}'.format(action, str(self.policy[s, a][0])))
			print('}')
			
	def get_actual_action_values(self):
		print('Action values after policy improvement are:')
		for s, state in enumerate(self.states):
			print('Action values for state {}:\n'.format(state) + '{')
			for a, action in enumerate(self.actions):
				print('    Action: {} --> Value: {}'.format(action, str(self.action_values[s, a][0])))
	
	def policy_iteration (self, gamma=1, theta=0.001):
		"""
		:float gamma: 0 (myopic) <= gamma <= 1 (infinitely farsighted)
		:float theta: << 1 values update delta in order to stop value iteration
		:return:
		"""
		self.gamma = gamma
		self.theta = theta
		counter = 0
		while self.train:
			print('Started training iteration No. ' + str(counter))
			self.state_values = self.valAct.policy_evaluation(self)
			print('    State values after policy evaulation are: ' + str(self.state_values))
			self.action_values = self.polAct.policy_improvement(self)
			self.get_actual_action_values()
			self.get_actual_policy()
			counter += 1
		
		print("\nFinished training after " + str(counter) + " iterations!")
