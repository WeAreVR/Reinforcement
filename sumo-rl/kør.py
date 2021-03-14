import gym
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
import traci

import numpy as np                       
from keras.models import Sequential                      
from keras.layers import Dense, Activation, Flatten                       
from keras.optimizers import Adam                                           
from rl.agents.dqn import DQNAgent                       
from rl.policy import BoltzmannQPolicy                       
from rl.memory import SequentialMemory


if __name__ == '__main__':

    env = SumoEnvironment(net_file='nets/double/network.net.xml',
                                    route_file='nets/double/flow.rou.xml',
                                    out_csv_name='outputs/Data/dqn',
                                    single_agent=True,
                                    use_gui=True,
                                    num_seconds=10000,
                                    max_depart_delay=0)
    
    nb_actions = env.action_space.n                                               
    model = Sequential()                       
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))                      
    model.add(Dense(16))                       
    model.add(Activation('relu'))                       
    model.add(Dense(16))                       
    model.add(Activation('relu'))                       
    model.add(Dense(16))                      
    model.add(Activation('relu'))                       
    model.add(Dense(nb_actions))                       
    model.add(Activation('linear'))                       
    print(model.summary())

    memory = SequentialMemory(limit=10000, window_length=1)                       
    policy = BoltzmannQPolicy()  
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,                                      
        target_model_update=1e-2, policy=policy)  
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])  
    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)            
    dqn.test(env, nb_episodes=5, visualize=False)                




                                    

    