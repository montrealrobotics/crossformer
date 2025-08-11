import collections
import numpy as np
import torch
import flax
import jax

from crossformer.data.dataset import make_single_dataset

from ml_collections import  ConfigDict, config_flags

from crossformer.model.crossformer_model import CrossFormerModel

def stack_and_pad(history: collections.deque, num_obs: int):
    ## copied from scripts.server.py

    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs

class CrossFormerQudaruped:


    def __init__(self, model_path, model_step, horizon = 1, app_prev_act = True):
        self.rng = jax.random.PRNGKey(0) ## I am not sure if we should use args.seed
        self.model = CrossFormerModel.load_pretrained(model_path, step= model_step)
        self.proprio_normalization_statistics = self.model.dataset_statistics["proprio_quadruped"]
        self.unnormalization_statistics = self.model.dataset_statistics['action']

        #create history 
        self.history = collections.deque(maxlen=horizon)
        self.obs_dim =  self.proprio_normalization_statistics['mean'].shape[-1]

        
        self.task = self.model.create_tasks(texts=['walk'])
        self.prev_act = np.zeros((12), dtype=np.float32)
        self.app_prev_act = app_prev_act
        self.horizon = horizon
        self.num_obs = 0

    def get_config(self):
        flat_config = flax.traverse_util.flatten_dict(
            self.model.config, keep_empty_nodes=True
        )
        config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
        return config


    def reset(self):
        self.observation_cue = np.zeros((1, self.horizon, self.obs_dim), dtype=np.float32)
        self.timestep_mask_cue = np.zeros((1, self.horizon), dtype=np.float32)
        self.task = self.model.create_tasks(texts=['walk'])
        self.prev_act = np.zeros((12), dtype=np.float32)
        self.num_obs = 0


    def __call__(self, obs):
        return self.get_action(obs)


    def get_action(self, obs):
        #NOTE: this only works for a batch_size = 1 setting
                
        prop_dim = 45 #this is magic number
        
        #need to shorten it
        obs = obs[:, :prop_dim]
        if self.app_prev_act:
            obs = np.concatenate((obs, self.prev_act[None, :]), axis = -1)
        mu = self.proprio_normalization_statistics['mean']
        std = self.proprio_normalization_statistics['std']

        #so it is window x dims now
        obs = obs[0]
        obs = (obs - mu) / (std + 1e-8)

        element = {
            "proprio_quadruped": obs,
        }
        self.history.append(element)
        self.num_obs += 1

        element = {
            "proprio_quadruped": obs,
        }
        self.history.append(element)
        self.num_obs += 1
        obs = stack_and_pad(self.history, self.num_obs)
        #add batch dimension
        obs = jax.tree_map(lambda x: x[None], obs)

        self.rng, key = jax.random.split(self.rng)
        actions = self.model.sample_actions(
            obs,
            self.task, #task is "walk"
            unnormalization_statistics = self.unnormalization_statistics,
            head_name="quadruped",
            rng=self.rng,
        )
        action = actions[0][0]
        action = np.array(action)
        self.prev_act = action  #set it before transforming
        action = torch.from_numpy(action).float()
        # Step environment

        action = action.unsqueeze(0)

        return action

def main(_):
    import tqdm 
    import copy

    model = CrossFormerQudaruped(model_path = FLAGS.config.pretrained_path, model_step = None)
    proprio_normalization_statistics = model.model.dataset_statistics["proprio_quadruped"]
    unnormalization_statistics = model.model.dataset_statistics['action']

    config = model.get_config()
    rng = jax.random.PRNGKey(0) ## I am not sure if we should use args.seed
    #this normalizes the actions & observations 
    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=False,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )

    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        print("batch errors")
        batch = next(train_data_iter)

        obs = batch['observation']
        proprio = obs['proprio_quadruped']
        #normalizing observations....seems to have less error, not sure what that is about
        obs_norm = copy.deepcopy(obs)
        obs_norm['proprio_quadruped'] = (proprio - proprio_normalization_statistics['mean']) / (proprio_normalization_statistics['std'] + 1e-8)
        #denormalizing made things worse (as it should because the values would be wrong)
        obs_denorm = copy.deepcopy(obs)
        obs_denorm['proprio_quadruped'] = (proprio * (proprio_normalization_statistics['std'] + 1e-8)) + proprio_normalization_statistics['mean']

        task = batch['task']
        task = model.model.create_tasks(texts=task['language_instruction'])
        true_act = batch['action']

        # List of observation variants to test
        obs_variants = [
            ("obs", obs),
            ("obs_denorm", obs_denorm),
            ("obs_norm", obs_norm)
        ]

        for obs_name, obs_variant in obs_variants:
            print(obs_name)
            #when using the dataset w/ the finetune configs, unnormalizing does not make sense to do
            action = model.model.sample_actions(
                obs_variant,
                task,
                unnormalization_statistics=unnormalization_statistics,
                head_name="quadruped",
                rng=rng
            )
            action_no_unnorm = model.model.sample_actions(
                obs_variant,
                task,
                unnormalization_statistics=None, #should be None if comparing to normalized actions
                head_name="quadruped",
                rng=rng
            )
            #just tellsus how much normalizing is making adifference
            #error_action = np.linalg.norm(np.array(action) - np.array(action_no_unnorm), axis=-1)
            #print(f"Error between actions ({obs_name}) with and without unnormalization_statistics: {error_action.mean()}")

            error_action_true = np.linalg.norm(np.array(action) - np.array(true_act), axis=-1)
            print(f"Error between action ({obs_name}, with unnormalization_statistics) and true_act: {error_action_true.mean()}")

            error_action_no_unnorm_true = np.linalg.norm(np.array(action_no_unnorm) - np.array(true_act), axis=-1)
            print(f"Error between action ({obs_name}, without unnormalization_statistics) and true_act: {error_action_no_unnorm_true.mean()}")
            print('\n')

if __name__ == "__main__":
    from absl import flags, app
    import os
    FLAGS = flags.FLAGS

    flags.DEFINE_string("name", "experiment", "Experiment name.")
    flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

    default_config_file = os.path.join(
        os.path.dirname(__file__), "configs/finetune_config.py"
    )
    config_flags.DEFINE_config_file(
        "config",
        default_config_file,
        "File path to the training hyperparameter configuration.",
        lock_config=False,
    )
    app.run(main) 