import argparse
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
from stable_baselines3 import DQN
from keras.models import Sequential                      
from keras.layers import Dense, Activation, Flatten                       
from keras.optimizers import Adam                                           
from rl.agents.dqn import DQNAgent                       
from rl.policy import BoltzmannQPolicy                       
from rl.memory import SequentialMemory


if __name__ == '__main__':
    try:
        prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
        prs.add_argument("-route", dest="route", type=str, default='nets/2way-single-intersection/single-intersection-vhvh.rou.xml', help="Route definition xml file.\n")
        prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
        prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
        prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
        prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
        prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
        prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
        prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
        prs.add_argument("-gui", action="store_true", default=True, help="Run with visualization on SUMO.\n")
        prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
        prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
        prs.add_argument("-r", dest="reward", type=str, default='wait', required=False, help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n")
        prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
        args = prs.parse_args()
    except BaseException:
        print(prs)
    out_csv = 'outputs/Data/dqn'
   
    env = SumoEnvironment(net_file='nets/double/network.net.xml',
                        single_agent=True,
                        route_file='nets/double/flow.rou.xml',
                        out_csv_name='outputs/Data/dqn',
                        use_gui=True,
                        num_seconds=80000,
                        max_depart_delay=0)                                
                                    
    '''env = SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          use_gui=args.gui, 
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          max_depart_delay=0)                                
    '''
   
    
    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.01,
        learning_starts=0,
        train_freq=1,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1
    )
    model.learn(total_timesteps=80000)       
    print("vi er nu her")
    obs = env.reset()
    for i in range(80000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()   
    



                                    

    