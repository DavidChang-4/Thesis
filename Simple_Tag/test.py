from pettingzoo.mpe import simple_tag_v3

n_evader=1 # default 1
n_pursuer = 5 # default 3
n_obstacle = 2 # default 2
max_cycles = 25 # default 25

render_mode = "" # '' or 'human'

env = simple_tag_v3.env(num_good=n_evader, num_adversaries=n_pursuer, num_obstacles=n_obstacle, render_mode=render_mode)
env.reset(seed=42)

for agent in env.agent_iter():
    # Observations: [vel, pos, landmark_rel_pos, other_rel_pos, other_vel]
    # Action: [none, left, right, down, up]
    # Agents: [pursuers, evaders]
    observation, reward, termination, truncation, info = env.last()



    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()