import numpy as np

data_file_name = (
    'dmfd/data/ClothFoldRobotHard_numvariations1_eps1000_image_based_trajs.pkl'
)
data = np.load(data_file_name, allow_pickle=True)
num_eps = len(data['ob_img_trajs'])
print(f'Loaded {num_eps} episodes from {data_file_name}')

all_ep_data = []
for i in range(num_eps):
    ep_data = {
        'obs_img': data['ob_img_trajs'][i],
        'language_instruction': 'fold cloth along diagonal',
        'action': data['action_trajs'][i],
        'reward': data['total_normalized_performance'][i],
    }
    all_ep_data.append(ep_data)

## Create train/val split
num_train_eps = int(0.8 * num_eps)
num_val_eps = num_eps - num_train_eps

# shuffle episodes and get train/val splits
np.random.shuffle(all_ep_data)
train_ep_idx = [i for i in range(num_train_eps)]
val_ep_idx = [i for i in range(num_train_eps, num_eps)]

## Save these splits
# save each train episode separately, with the episode index in the file name
for ep_index in train_ep_idx:
    ep_file_name = f'dmfd/data/train/episode_{ep_index}.npy'
    np.save(ep_file_name, all_ep_data[ep_index], allow_pickle=True)
    print(f'Saved episode {ep_index} to {ep_file_name}')
print(f'Saved {num_train_eps} episodes for training')

# save each val episode separately, with the episode index in the file name
for ep_index in val_ep_idx:
    ep_file_name = f'dmfd/data/val/episode_{ep_index}.npy'
    np.save(ep_file_name, all_ep_data[ep_index], allow_pickle=True)
    print(f'Saved episode {ep_index} to {ep_file_name}')
print(f'Saved {num_val_eps} episodes for validation')

print('Done!')
