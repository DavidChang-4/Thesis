# %%
###############################
# Imports
###############################
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential


# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
import tempfile

# CPU/GPU
is_fork = multiprocessing.get_start_method() == "fork"
device = (torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu"))
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)
print(device)

# %%
###############################
# Hyperparams
###############################

# Sampling (frames per batch)
frames_per_batch = 6_000  # Number of team frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training (batch size)
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

# %%
###############################
# Environment Setup
###############################
max_steps = 100  # Steps before 'done' is automatically set.
num_vmas_envs = (frames_per_batch // max_steps)  # Leverage batch simulation. 
scenario_name = "simple_tag"

# Num pursuers/evaders
n_pursuers = 2
n_evaders = 1
n_obstacles = 2
render = "human" # '' or 'human'

base_env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=True,
    max_steps=max_steps,
    device=vmas_device,

    num_good_agents=n_evaders,
    num_adversaries=n_pursuers,
    num_landmarks=n_obstacles,
    render_mode=render,
)

# %%
###############################
# Printing environment keys
###############################
# All specs have leading shape (num_vmas_envs, n_agents) except for 'done.'
# print("action_spec:", env.full_action_spec)
# print("reward_spec:", env.full_reward_spec)
# print("done_spec:", env.full_done_spec)
# print("observation_spec:", env.observation_spec)

# print("action_keys:", env.action_keys)
print("reward_keys:", env.reward_keys)
# print("done_keys:", env.done_keys)
print(base_env.full_action_spec[('adversary', 'action')])

# %%
###############################
# Modify environment
# Group pursuer rewards
###############################
env = TransformedEnv(
    base_env,
    RewardSum(
        in_keys=base_env.reward_keys,
        # out_keys=[("adversaries", "episode_reward"), ("agents", "episode_reward")], TODO: necessary?
        reset_keys=["_reset"] * len(base_env.group_map.keys()),
    )
)
check_env_specs(env) # Check env setup

# %%
# ###############################
# # Rollout
# ###############################
# with torch.no_grad():
#    env.rollout(
#        max_steps=max_steps,
#        #policy=policy,
#        callback=lambda env, _: env.render(),
#        auto_cast_to_device=True,
#        break_when_any_done=False,
#    )

# %%
print(env.observation_spec["agent", "observation"].shape[-1])

# %%
###############################
# Policy Network
# Returns policies[group]
###############################

# 1) Create MLP agents for adversaries
policy_modules = {}
for group, agents in env.group_map.items():
    share_parameters_policy = True

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],  # n_obs_per_agent
            n_agent_outputs=2 * env.full_action_spec[group, "action"].shape[-1],  # 2 * n_actions_per_agents
            n_agents=len(agents),
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    )

    # 2) Wrap NN -> TensorDictModule
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "loc"), (group, "scale")], # TODO: maybe outkeys needs to be 1 param
    )

    policy_modules[group] = policy_module # Group = 'adversary' or 'agent'




# 3) Create ProbabilisticActor from policy networks
policies = {}
for group, _agents in env.group_map.items():
    policy = ProbabilisticActor(
        module=policy_modules[group],
        spec=env.full_action_spec[group, "action"],
        in_keys=[(group, "loc"), (group, "scale")],
        out_keys=[(group, "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec[group, "action"].space.low,
            "high": env.full_action_spec[group, "action"].space.high,
        },
        return_log_prob=True,
        log_prob_key=(group, "sample_log_prob"),
    )  # we'll need the log-prob for the PPO loss
    policies[group] = policy


# %%
# ###############################
# # Critic Network: NO CAT!!!!!!!!!!!!!!
# # Returns policy_modules[], policies[], and exploration_policies[]
# ###############################
# share_parameters_critic = True
# mappo = True  # IPPO if False

# critics = {}
# for group, agents in env.group_map.items():

#     # TODO: do i need cat module
#     critic_net = MultiAgentMLP(
#         n_agent_inputs=env.observation_spec[group, "observation"].shape[-1] + env.full_action_spec[group, "action"].shape[-1],
#         n_agent_outputs=1,  # 1 value per agent
#         n_agents=len(agents),
#         centralised=mappo,
#         share_params=share_parameters_critic,
#         device=device,
#         depth=2,
#         num_cells=256,
#         activation_class=torch.nn.Tanh,
#     )

#     critic = TensorDictModule(
#         module=critic_net,
#         in_keys=[(group, "obs_action")],
#         out_keys=[(group, "state_action_value")],
#     )

#     critics[group] = critic

# %%
###############################
# Critic Network
# Returns policy_modules[], policies[], and exploration_policies[]
###############################
share_parameters_critic = True
mappo = True  # IPPO if False

critics = {}
for group, agents in env.group_map.items():

    cat_module = TensorDictModule(
        lambda obs, action: torch.cat([obs, action], dim=-1),
        in_keys=[(group, "observation"), (group, "action")],
        out_keys=[(group, "obs_action")],
    )

    # TODO: do i need cat module
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[group, "observation"].shape[-1] + env.full_action_spec[group, "action"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=len(agents),
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic_module = TensorDictModule(
        module=critic_net,
        in_keys=[(group, "obs_action")],
        out_keys=[(group, "state_value")],
    )

    critics[group] = TensorDictSequential(
        cat_module, critic_module
    )

# %%
# # Reset the environment once
# env_state = env.reset()

# # Loop over groups to apply policies and critics
# for group in env.group_map:
#     try:
#         # Apply the policy for the current group
#         policy_output = policies[group](env_state)
#         print(f"Running policy for group {group}:", policy_output)
        
#         # Apply the critic for the current group
#         critic_output = critics[group](env_state)
#         print(f"Running value for group {group}:", critic_output)
#     except Exception as e:
#         print(f"Error for group {group}: {e}")


# %%
# NO CAT!!!!!!!!!!!!!!
# env_state = env.reset()

# # Concatenate observation and action into obs_action
# for group in env.group_map:
#     observation = env_state.get((group, 'observation'))
#     action = env_state.get((group, 'action'), torch.zeros_like(observation))  # Default to zeros if action is missing
#     env_state.set((group, 'obs_action'), torch.cat([observation, action], dim=-1))

# %%
# Run policy example
reset_td = env.reset()
for group, _agents in env.group_map.items():
    print(
        f"Running value and policy for group '{group}':",
        critics[group](policies[group](reset_td)),
    )

# %%
###############################
# Data Collector
###############################
agents_policy = TensorDictSequential(*policies.values())
print(agents_policy)
collector = SyncDataCollector(
    env,
    agents_policy, # TODO: uh lol
    device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

###############################
# Replay Buffer
###############################
replay_buffers = {}
for group, _agents in env.group_map.items():
    replay_buffer = ReplayBuffer(
        # TODO: lazytensor or lazymemmap
        storage=LazyTensorStorage(frames_per_batch, device=device),  # We will store up to memory_size multi-agent transitions
        sampler=SamplerWithoutReplacement(), 
        batch_size=minibatch_size,  # Sample this size
    )
    replay_buffers[group] = replay_buffer

# %%
###############################
# Loss Function
###############################
optimisers = {}
for group, _agents in env.group_map.items():
    loss_module = ClipPPOLoss(
        actor_network=policies[group],  # Use the non-explorative policies
        critic_network=critics[group],
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        sample_log_prob=(group, "sample_log_prob"),
        value=(group, "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=(group, "done"),
        terminated=(group, "terminated"),
    )


    loss_module.make_value_estimator(
        ValueEstimators.GAE, # TD0 VS GAE
        gamma=gamma, 
        lmbda=lmbda
    ) 
    GAE = loss_module.value_estimator # we build GAE TODO: do for each agent/adv?

    optimisers[group] = torch.optim.Adam(loss_module.parameters(), lr)

# %%
# Train

# pbar = tqdm(
#     total=n_iters,
#     desc=", ".join(
#         [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
#     ),
# )
episode_reward_mean_map = {group: [] for group in env.group_map.keys()}

train_group_map = copy.deepcopy(env.group_map)

print(len(collector))
for tensordict_data in collector:
    # print(tensordict_data)
    print("OK")
    # tensordict_data.set(
    #     ("next", "agents", "done"),
    #     tensordict_data.get(("next", "done"))
    #     .unsqueeze(-1)
    #     .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    # )
    # tensordict_data.set(
    #     ("next", "agents", "terminated"),
    #     tensordict_data.get(("next", "terminated"))
    #     .unsqueeze(-1)
    #     .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    # )
    # # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

    # with torch.no_grad():
    #     GAE(
    #         tensordict_data,
    #         params=loss_module.critic_network_params,
    #         target_params=loss_module.target_critic_network_params,
    #     )  # Compute GAE and add it to the data

    # data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    # replay_buffer.extend(data_view)

    # for _ in range(num_epochs):
    #     for _ in range(frames_per_batch // minibatch_size):
    #         subdata = replay_buffer.sample()
    #         loss_vals = loss_module(subdata)

    #         loss_value = (
    #             loss_vals["loss_objective"]
    #             + loss_vals["loss_critic"]
    #             + loss_vals["loss_entropy"]
    #         )

    #         loss_value.backward()

    #         torch.nn.utils.clip_grad_norm_(
    #             loss_module.parameters(), max_grad_norm
    #         )  # Optional

    #         optim.step()
    #         optim.zero_grad()

    # collector.update_policy_weights_()

    # # Logging
    # done = tensordict_data.get(("next", "agents", "done"))
    # episode_reward_mean = (
    #     tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    # )
    # episode_reward_mean_list.append(episode_reward_mean)
    # pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    # pbar.update()


