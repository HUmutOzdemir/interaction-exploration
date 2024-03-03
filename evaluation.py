import torch.multiprocessing as mp
import random
import numpy as np

from interaction_exploration.config import get_config
from interaction_exploration.run import get_trainer
from interaction_exploration.trainer import *
from interaction_exploration.models.policy import * 

from rl.common.utils import logger
from rl.common.env_utils import construct_envs, get_env_class

mp.set_start_method('spawn')

config = 'interaction_exploration/config/intexp.yaml'
options = [
    'ENV.NUM_STEPS', '256',
    'NUM_PROCESSES', '1',
    'EVAL.DATASET', 'interaction_exploration/data/test_episodes_K_16.json',
    'TORCH_GPU_ID', '0',
    'X_DISPLAY', ':0',
    'CHECKPOINT_FOLDER', 'models/eval',
    'LOAD', 'models/ckpt.48.pth',
    'MODEL.BEACON_MODEL', 'models/epoch=04-val_loss=0.4979.ckpt'
]

config = get_config(config, opts=options)

random.seed(config.SEED)
np.random.seed(config.SEED)

trainer = get_trainer(config)

trainer.init_viz()
test_episodes = ['FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230']

trainer.config.defrost()
trainer.config.ENV.TEST_EPISODES = test_episodes
trainer.config.ENV.TEST_EPISODE_COUNT = len(test_episodes)
trainer.config.NUM_PROCESSES = 1
trainer.config.MODE = 'eval'
trainer.config.freeze()

checkpoint_path = trainer.config.LOAD
ckpt_dict = trainer.load_checkpoint(checkpoint_path, map_location="cpu")
ppo_cfg = trainer.config.RL.PPO

logger.info(f"env config: {trainer.config}")

trainer.envs = construct_envs(trainer.config, get_env_class(trainer.config.ENV.ENV_NAME))
trainer._setup_actor_critic_agent(ppo_cfg)

logger.info(checkpoint_path)
logger.info(f"num_steps: {trainer.config.ENV.NUM_STEPS}")

trainer.agent.load_state_dict(ckpt_dict["state_dict"])
trainer.actor_critic = trainer.agent.actor_critic

observations = trainer.envs.reset()
batch = trainer.batch_obs(observations, trainer.device)

current_episode_reward = torch.zeros(
    trainer.envs.num_envs, 1, device=trainer.device
)

test_recurrent_hidden_states = torch.zeros(
    trainer.actor_critic.net.num_recurrent_layers,
    trainer.config.NUM_PROCESSES,
    ppo_cfg.hidden_size,
    device=trainer.device,
)
prev_actions = torch.zeros(
    trainer.config.NUM_PROCESSES, 1, device=trainer.device, dtype=torch.long
)
not_done_masks = torch.zeros(
    trainer.config.NUM_PROCESSES, 1, device=trainer.device
)
stats_episodes = dict()  
trainer.actor_critic.eval()