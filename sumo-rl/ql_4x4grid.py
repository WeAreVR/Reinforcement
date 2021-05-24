import argparse
import os
import sys
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 3

    env = SumoEnvironment(net_file='nets/2way-single-intersection/single-intersection.net.xml',
                          route_file='nets/2way-single-intersection/single-intersection-gen.rou.xml',
                          out_csv_name='outputs/ql/ql',
                          use_gui=False,
                          num_seconds=80000,
                          yellow_time=4,
                          min_green=5,
                          max_green=120,
                          max_depart_delay=300)

    for run in range(1, runs+1):
        initial_states = env.reset()
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) for ts in env.ts_ids}
        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)

            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        env.close()


