from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Dmfd(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for DMfD dataset.

    Pick place action space for cloth folding diagonal task.
    """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(32, 32, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Image observation of cloth.',
                        ),
                        # 'wrist_image': tfds.features.Image(
                        #     shape=(64, 64, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Wrist camera RGB observation.',
                        # ),
                        # 'state': tfds.features.Tensor(
                        #     shape=(10,),
                        #     dtype=np.float32,
                        #     doc='Robot state, consists of [7x robot joint angles, '
                        #         '2x gripper position, 1x door opening angle].',
                        # )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(4,),
                        dtype=np.float32,
                        doc='Robot action, consists of x,y,z goal and picker command'
                         'picker<0.5 = open, picker>0.5 = close.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward as a normalized performance metric in [0, 1].'
                        '0 = no change from initial state. 1 = perfect fold.'
                        '-ve performance means the cloth is worse off than initial state.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)[()]
            # print(f'Loaded episode from {episode_path}')

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            num_steps = len(data['obs_img'])
            # for i, step in enumerate(data):
            for i in range(num_steps):
                # print(f'Processing step {i} of {num_steps}')
                # import ipdb; ipdb.set_trace()
                # compute Kona language embedding
                # try this, if fails then load ipdb and debug
                try:
                    language_embedding = self._embed([data['language_instruction']])[0].numpy()
                except:
                    import ipdb; ipdb.set_trace()
                    language_embedding = self._embed([data['language_instruction']])[0].numpy()

                episode.append({
                    'observation': {
                        'image': data['obs_img'][i],
                    },
                    'action': np.array(data['action'][i], dtype=np.float32),
                    'discount': 1.0,
                    'reward': 0.0 if i < (num_steps - 1) else data['reward'],
                    'is_first': i == 0,
                    'is_last': i == (num_steps - 1),
                    'is_terminal': i == (num_steps - 1),
                    'language_instruction': data['language_instruction'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

