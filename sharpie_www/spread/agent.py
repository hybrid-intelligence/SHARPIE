import numpy as np 
from tensorflow.keras.layers import Dense

class ActionValueNetwork:
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_actions = network_config.get("num_actions")
        self.step_size=network_config.get('step_size')
    def create_model(self):
        i = Input(shape=self.state_dim)
        x = Dense(256, activation='relu')(i)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.num_actions, activation='linear')(x)
        model = Model(i, x)
        model.compile(optimizer=Adam(lr=self.step_size),loss='mse')
        return model
    

epsilon = 1  
EPSILON_DECAY = 0.998 
MIN_EPSILON = 0.01

agent_info = {
             'network_config': {
                 'state_dim': 21,
                 'num_actions': 5,
                 'step_size':1e-3
             },
             'replay_buffer_size': 256,
             'minibatch_sz': 16,
             'num_replay_updates_per_step': 2,
             'gamma': 0.99,
             'seed': 0}

class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
       
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state):
     
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
       
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)
    

    
class Agent:
    def __init__(self, agent_config):
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], 
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config'])
        
        self.model=self.network.create_model()
        
        self.target_model=self.network.create_model()
        
        self.num_actions = agent_config['network_config']['num_actions']
        
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        self.last_states = None
        self.actions = None
        self.epsilon = epsilon
        self.sum_rewards = {'agent_0':0,'agent_1':0,'agent_2':0}
        self.episode_steps = 0
   
    def agent_start(self):
        self.sum_rewards = {'agent_0':0,'agent_1':0,'agent_2':0}
        self.episode_steps=0
        self.last_states=env.reset()
        for i,m in enumerate(self.last_states.keys()):
            self.last_states[m]=np.array([np.append(self.last_states[m],b[i])])
        self.actions = {agent: self.policy(agent,self.last_states[agent]) for agent in env.agents}
        actions=self.actions
        return actions
    
    def agent_step(self,states,rewards,terminals):
        self.sum_rewards={agent: (self.sum_rewards[agent]+rewards[agent]) for agent in rewards}
        self.episode_steps += 1
        for i,m in enumerate(states.keys()):
            states[m]=np.array([np.append(states[m],b[i])])
        
        for agent in states:
            state=states[agent]
            last_state=self.last_states[agent]
            action=self.actions[agent]
            reward=rewards[agent]
            terminal=terminals[agent]
            self.replay_buffer.append(last_state,action, reward, terminal, state)
        
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.target_model.set_weights(self.model.get_weights())
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                self.agent_train(experiences)
        
        
      
        self.last_states=states
        self.actions = {agent: self.policy(agent,self.last_states[agent]) for agent in env.agents}
        actions=self.actions
        return actions
    
    def agent_train(self,experiences):
        
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        batch_size1 = states.shape[0]
        q_next_mat = self.target_model.predict(next_states)
        
        v_next_vec = np.max(q_next_mat, axis=1)*(1-terminals)
        
        target_vec = rewards + self.discount*v_next_vec
       
        q_mat = self.model.predict(states)
      
        batch_indices = np.arange(q_mat.shape[0])

        X=states
        q_mat[batch_indices,actions] = target_vec
 
        self.model.fit(X,q_mat,batch_size=batch_size1,verbose=0,shuffle=False)