# %%
import torch.multiprocessing as mp
import random
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import skimage.transform as st
from torchmetrics import Accuracy, CohenKappa, F1Score, JaccardIndex, Precision, Recall
from collections import Counter
import json

from interaction_exploration.config import get_config
from interaction_exploration.run import get_trainer
from interaction_exploration.trainer import *
from interaction_exploration.models.policy import * 

from rl.common.utils import logger
from rl.common.env_utils import construct_envs, get_env_class

# %%
def get_gt_affordance(last_event, size=(2, 128, 128)):
    affordance_types = [['pickupable'], ['moveable', 'pickupable']]

    gt_affordances = np.zeros((2, 300, 300), dtype=np.float32)
    for obj in last_event.metadata["objects"]:
        if obj["objectId"] not in last_event.instance_masks:
            continue
        # Extract Object Instance Mask
        obj_instance_seg = last_event.instance_masks[obj["objectId"]]
        # Extrach GT Object Affordances
        obj_gt_affordance = np.array(
            [np.any([obj[a] for a in aff]) for aff in affordance_types]
        )
        # Write Object Affordance into the Affordance Map
        gt_affordances[:, obj_instance_seg] = obj_gt_affordance[:, None].astype(
            np.float32
        )
    
    if size:
        gt_affordances = st.resize(gt_affordances, size, order=0, preserve_range=True, anti_aliasing=False)
    
    return gt_affordances

# %%
def calculate_affordance_evaluation_metrics(last_event, estimation, scores=None, _threshold = 0.5, _affordance_types = ['pickupable', 'moveable_pickupable'], size = (128, 128)):

    metrics = {
        "IoU": JaccardIndex(2),
        "accuracy": Accuracy(2),
        "cohen_kappa": CohenKappa(2),
        "precision": Precision(),
        "recall": Recall(),
        "f1_score": F1Score(),
        "true_positive": lambda est, gt: torch.sum(
            torch.logical_and(est > _threshold, gt.type(torch.bool))
        ),
        "true_negative": lambda est, gt: torch.sum(
            torch.logical_and(est < _threshold, gt.type(torch.bool))
        ),
        "false_positive": lambda est, gt: torch.sum(
            torch.logical_and(est > _threshold, gt.type(torch.bool))
        ),
        "false_negative": lambda est, gt: torch.sum(
            torch.logical_and(est < _threshold, gt.type(torch.bool))
        ),
    }

    ground_truth = get_gt_affordance(last_event)
    if scores == None:
        scores = {
            affordance_type: {m: [] for m in metrics}
            for affordance_type in _affordance_types
        }
        for affordance_type in _affordance_types:
            scores[affordance_type]["object_level_accuracy"] = []

    for m in metrics:
        for est, gt, affordance_type in zip(
            estimation, ground_truth, _affordance_types
        ):
            result = metrics[m](torch.from_numpy(est.cpu().numpy()), torch.from_numpy(gt).type(torch.int32)).item()
            if np.isfinite(result):
                scores[affordance_type][m].append(result)

    estimation_npy = (estimation.cpu().numpy() > _threshold).astype(int)
    ground_truth_npy = ground_truth.astype(int)
    gt_segmentation = last_event.instance_masks

    for est, gt, affordance_type in zip(estimation_npy, ground_truth_npy, _affordance_types):
        accuracies = []
        for class_ in gt_segmentation:
            if size:
                gt_segmentation_ = st.resize(gt_segmentation[class_], size, order=0, preserve_range=True, anti_aliasing=False)
                if gt_segmentation_.sum() == 0:
                    continue
            est_class = Counter(est[gt_segmentation_].flatten()).most_common(1)[0][0]
            gt_class = Counter(gt[gt_segmentation_].flatten()).most_common(1)[0][0]
            accuracies.append(float(gt_class == est_class))
        scores[affordance_type]['object_level_accuracy'].append(np.mean(accuracies))

    return scores

# %%
def execute_single_episode(trainer, ppo_cfg, batch, current_episode_reward, test_recurrent_hidden_states, prev_actions, not_done_masks, stats_episodes):
    infos = None
    affordance_scores = [None for _ in range(trainer.envs.num_envs)]
    for step in range(ppo_cfg.num_steps):
        # Apply Action
        current_episodes = trainer.envs.current_episodes()

        with torch.no_grad():
            (
                _,
                actions,
                _,
                test_recurrent_hidden_states,
            ) = trainer.actor_critic.act(
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=False,
            )

            prev_actions.copy_(actions)

        outputs = trainer.envs.step([a[0].item() for a in actions])

        # Log Metrics
        observations, rewards, dones, infos = [
            list(x) for x in zip(*outputs)
        ]
        batch = trainer.batch_obs(observations, trainer.device)

        for i in range(trainer.envs.num_envs):
            last_event = trainer.envs.call_at(i, 'last_event')
            affordance_est = batch['aux'][i] * 0.06 + 0.04
            affordance_scores[i] = calculate_affordance_evaluation_metrics(last_event, batch['aux'][i], affordance_scores[i])

        not_done_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=trainer.device,
        )

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=trainer.device
        ).unsqueeze(1)
        current_episode_reward += rewards
        n_envs = trainer.envs.num_envs
        for i in range(n_envs):
            # episode ended
            if not_done_masks[i].item() == 0:
                episode_stats = dict()
                episode_stats["reward"] = current_episode_reward[i].item()
                episode_stats.update(
                    trainer._extract_scalars_from_info(infos[i])
                )
                for affordance in affordance_scores[i]:
                    for m in affordance_scores[i][affordance]:
                        episode_stats[f"{affordance}_{m}"] = np.mean(affordance_scores[i][affordance][m])
                current_episode_reward[i] = 0
                stats_episodes[
                    (
                        current_episodes[i]['scene_id'],
                        current_episodes[i]['episode_id'],
                    )
                ] = episode_stats

# %%
def execute_evaluation(trainer, ppo_cfg, num_episodes):
    stats_episodes = dict()

    for _ in tqdm(range(num_episodes)):
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
        trainer.actor_critic.eval()

        execute_single_episode(
            trainer,
            ppo_cfg,
            batch, 
            current_episode_reward, 
            test_recurrent_hidden_states, 
            prev_actions, 
            not_done_masks, 
            stats_episodes
        )

    # Log info so far
    num_episodes = len(stats_episodes)
    aggregated_stats = dict()
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = {
            'mean': (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            ),
            'std': np.std([v[stat_key] for v in stats_episodes.values()])
        }
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v['mean']:.4f} ({num_episodes} episodes)")
        logger.info(f"episode {k} Std: {v['std']:.4f} ({num_episodes} episodes)")

    return aggregated_stats

# %%
mp.set_start_method('spawn')

# %%
config = 'interaction_exploration/config/intexp.yaml'
options = [
    'ENV.NUM_STEPS', '512',
    'NUM_PROCESSES', '4',
    'EVAL.DATASET', 'interaction_exploration/data/test_episodes_K_16.json',
    'TORCH_GPU_ID', '0',
    'X_DISPLAY', ':0',
    'CHECKPOINT_FOLDER', 'models/eval',
    'LOAD', 'interaction_exploration/cv/intexp/run_comparison_last/ckpt.48.pth',
    'MODEL.BEACON_MODEL', 'interaction_exploration/cv/intexp/run_comparison_last/unet/epoch=15-val_loss=0.5353.ckpt'
]

config = get_config(config, opts=options)
config

# %%
random.seed(config.SEED)
np.random.seed(config.SEED)

trainer = get_trainer(config)

# %%
trainer.init_viz()
test_episodes = ['FloorPlan226', 'FloorPlan227', 'FloorPlan228', 'FloorPlan229', 'FloorPlan230']

trainer.config.defrost()
trainer.config.ENV.TEST_EPISODES = test_episodes
trainer.config.ENV.TEST_EPISODE_COUNT = len(test_episodes)
trainer.config.MODE = 'train'
trainer.config.freeze()

checkpoint_path = trainer.config.LOAD
ckpt_dict = trainer.load_checkpoint(checkpoint_path, map_location="cpu")
ppo_cfg = trainer.config.RL.PPO

logger.info(f"env config: {trainer.config}")

# %%
trainer.envs = construct_envs(trainer.config, get_env_class(trainer.config.ENV.ENV_NAME))
trainer._setup_actor_critic_agent(ppo_cfg)

# %%
logger.info(checkpoint_path)
logger.info(f"num_steps: {trainer.config.ENV.NUM_STEPS}")

trainer.agent.load_state_dict(ckpt_dict["state_dict"])
trainer.actor_critic = trainer.agent.actor_critic

# %%
results = execute_evaluation(trainer, ppo_cfg, 100)

# %%
with open(f"{config.CHECKPOINT_FOLDER}/results.json", "w") as outfile:
    json.dump(results, outfile, indent=4, sort_keys=False)


