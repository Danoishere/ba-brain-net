class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []