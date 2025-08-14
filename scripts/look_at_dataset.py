from crossformer.data.dataset import make_single_dataset
from ml_collections import ConfigDict
import tensorflow as tf

dataset_kwargs = ConfigDict({
    'name': 'flat_quadruped_dataset',
    'data_dir': "/home/mprzystupa/tensorflow_datasets", #'/home/mila/m/michael.przystupa/scratch/tensorflow_datasets',
    'image_obs_keys': {},  # no images used
    'proprio_obs_keys': {'quadruped': 'state'},
    'language_key': 'language_instruction',
    'action_proprio_normalization_type': 'normal',
    'action_normalization_mask': [True, True, True, True, True, True, True, True, True, True, True, True],
    'skip_norm_keys': [],
    # 'standardize_fn': ModuleSpec.create("crossformer.data.oxe.oxe_standardization_transforms:quadruped_dataset_transform"),
    # Uncomment and set these if needed for performance:
    # 'num_parallel_reads': 8,
    # 'num_parallel_calls': 16,
})

traj_transform_kwargs = ConfigDict({
    'window_size': 1,
    'action_horizon': 4,
    'goal_relabeling_strategy': 'uniform',
    'task_augment_strategy': 'delete_task_conditioning',
    'task_augment_kwargs': {'keep_image_prob': 0.5},
    # 'num_parallel_calls': 16,  # Uncomment if needed
})

workspace_augment_kwargs = {
    'random_resized_crop': {'scale': [0.8, 1.0], 'ratio': [0.9, 1.1]},
    'random_brightness': [0.1],
    'random_contrast': [0.9, 1.1],
    'random_saturation': [0.9, 1.1],
    'random_hue': [0.05],
    'augment_order': [
        'random_resized_crop',
        'random_brightness',
        'random_contrast',
        'random_saturation',
        'random_hue',
    ],
}

frame_transform_kwargs = ConfigDict({
    'resize_size': {
        'primary': (224, 224),
    },
    'image_augment_kwargs': {
        'primary': workspace_augment_kwargs,
    },
})

# Load dataset
dataset = make_single_dataset(
    dataset_kwargs,
    traj_transform_kwargs=traj_transform_kwargs,
    frame_transform_kwargs=frame_transform_kwargs,
    train=False,
)

# Create iterator and get a batch
train_data_iter = (
    dataset.repeat()
    #.unbatch()
    #.shuffle(1000)
    .batch(32)
    .iterator()
)
example_batch = next(train_data_iter)

# Print batch keys and shapes
def print_batch_info(batch):
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: {type(v)}")

print("Example batch:")
print_batch_info(example_batch)


