# Reinforcement
sumo-rl https://github.com/LucasAlegre/sumo-rl

Når RLlib A3C multiagent in a 4x4 grid: Køres nu med GUI

skal køres fra første folder indtil videre, ligges ind i filen kør.py, hvis ikke = ModuleNotFoundError: No module named 'sumo_rl.environment'

Single intersection skaber nederstående fejl
ModuleNotFoundError: No module named 'tensorflow.contrib'- skal vidst bruge tensorflow >2.0


stable-baselines3 DQN in a 2-way single intersection: Køres nu med GUI, men kører vidst kun på cpu

Den gemmer information efter ikke i reinforce mappen med i mappen før, ray_results, hvorfor ved jeg ikke

2 way køre til at starte med men laver denne fejl til sidst
FileNotFoundError: [Errno 2] No such file or directory: 'outputs/2way-single-intersection/dqn_run1.csv'


undersøg
WARNING:tensorflow:From C:\Users\tobia\anaconda3\envs\gpu_test\lib\site-packages\tensorflow\python\compat\v2_compat.py:96: disable_resource_variables
(from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.

check også https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control ikke lavet i openai gym though
