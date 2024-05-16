# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy 
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm

def sample_sequences(config):
    MAX_LOCATIONS = config['MAX_LOCATIONS']
    MAX_OBJECTS = config['MAX_OBJECTS']
    N_SEQS_TRAIN = config['N_SEQS_TRAIN']
    N_SEQS_TEST = config['N_SEQS_TEST']
    SAVE_DATA = config['SAVE_DATA']
    N_LOCATIONS = config['N_LOCATIONS']
    N_OBJECTS = config['N_OBJECTS']
    SZ_OBJECTIVE = (N_LOCATIONS, N_OBJECTS)
    N_STEPS = int(SZ_OBJECTIVE[0] * config['CAPACITY'])

    SPECIAL_TOKENS = ['[CLS]', '[UNK]', '[SEP]', '[MASK]', '[EMPTY]', '[PAD]', '[LAST]', '[PLACED]']

    id_to_token = []
    id_to_token += SPECIAL_TOKENS
    id_to_token += [f'{i}' for i in range(MAX_LOCATIONS)]
    id_to_token += [f'{i+MAX_LOCATIONS}' for i in range(MAX_OBJECTS)]

    token_to_id = {
        t: i for i, t in enumerate(id_to_token)
    }

    locations_start = len(SPECIAL_TOKENS)
    objects_start = locations_start + MAX_LOCATIONS


    dataroot = Path(f'datasets').joinpath(
        f'means_{config["N_MEANS"]}-locations_{config["N_LOCATIONS"]}-objects_{config["N_OBJECTS"]}-stddev_{config["STDDEV"]}'
    )

    train_objectives = torch.load(dataroot.joinpath('train_set.pt'))
    train_means = torch.load(dataroot.joinpath('train_mean_inds.pt'))
    test_close_objectives = torch.load(dataroot.joinpath('test_close_set.pt'))
    test_close_means = torch.load(dataroot.joinpath('test_close_mean_inds.pt'))
    test_far_objectives = torch.load(dataroot.joinpath('test_far_set.pt'))
    test_far_means = torch.load(dataroot.joinpath('test_far_mean_inds.pt'))
    all_means = torch.load(dataroot.joinpath('means.pt'))
    mu_far = torch.load(dataroot.joinpath('mu_far.pt'))
    X = torch.load(dataroot.joinpath('X.pt'))

    data_columns = [
        'primarySeqID', # this is the sequence number between 0 and seq_per_obj * n_objectives
        'datasetID', # this is in {train, test_close, test_far, test_close_pluso, test_far_pluso}
        'meanID', # this is the mean id
        'objectiveID', # this is the objective ID
        'seqID', # sequence id relative to the objective between 0 and seq_per_obj
        'stepID', # timestep within the sequence
        'objectAction', # human action
        'locationAction', # robot action
    ]
    data_columns += [f'location{i}' for i in range(MAX_LOCATIONS)]
    data_columns += [f'object{i}' for i in range(MAX_LOCATIONS)]

    locs_repr = np.eye(N_LOCATIONS)
    objs_repr = np.eye(N_OBJECTS)
    datasets = ['train', 'test_close', 'test_far']
    objectives = {
        'train': train_objectives, 
        'test_close': test_close_objectives, 
        'test_far': test_far_objectives,
    }
    mean_inds = {
        'train': train_means, 
        'test_close': test_close_means, 
        'test_far': test_far_means,
    }

    n_seqs = [N_SEQS_TRAIN, N_SEQS_TEST, N_SEQS_TEST]

    for dataset_i, dataset in enumerate(datasets):
        objective_set = objectives[dataset]
        n_seq = n_seqs[dataset_i]

        all_data = []
        sequences = torch.zeros((objective_set.shape[0] * n_seq * (N_STEPS +1),len(data_columns)), dtype=torch.long)
        with tqdm.tqdm(total=objective_set.shape[0]) as pbar:
            for objective_i, objective in enumerate(objective_set):
                for seq_i in range(n_seq):
                    t_seq_i = objective_i * n_seq + seq_i
                    mean_i = mean_inds[dataset][objective_i]

                    seql = N_STEPS
                    nobjs = N_OBJECTS
                    objects = np.random.choice(np.arange(nobjs), N_STEPS, replace=True)

                    ### Include which objects are in the scene, too
                    rmap = objs_repr[objects] @ objective.T @ locs_repr
                    rack = torch.ones(MAX_LOCATIONS, dtype = torch.long).view(MAX_LOCATIONS,1) * token_to_id['[EMPTY]']
                    rack[N_LOCATIONS:] = token_to_id['[PAD]']
                    object_occupancy = torch.ones((MAX_LOCATIONS,1), dtype=torch.long) * token_to_id['[PAD]']
                    object_occupancy[:len(objects)] = torch.tensor(objects+objects_start, dtype=torch.long).view(-1,1)
                    
                    list_ids = []
                    for step_i in range(seql+1):
                        step = torch.zeros((0,1), dtype=torch.long)
                        step = torch.cat((step, torch.tensor(t_seq_i).view(1,1)))
                        step = torch.cat((step, torch.tensor(dataset_i).view(1,1)))
                        step = torch.cat((step, torch.tensor(mean_i).view(1,1)))
                        step = torch.cat((step, torch.tensor(objective_i).view(1,1)))
                        step = torch.cat((step, torch.tensor(seq_i).view(1,1)))
                        step = torch.cat((step, torch.tensor(step_i).view(1,1)))

                        if step_i < min(N_STEPS, N_LOCATIONS):
                            # sample based preferences
                            # sm = nn.Softmax(-1)
                            # pval = sm(torch.tensor(rmap*10).view(-1)).view(rmap.shape[0],rmap.shape[1])
                            # object_list_id = np.random.choice(np.arange(len(objects)), p=pval.sum(1))
                            # location_id = np.random.choice(np.arange(N_LOCATIONS), p=sm(torch.tensor(rmap[object_list_id])))

                            # max based preferences
                            max_ind = np.random.choice(np.where((rmap == rmap.max()).reshape(-1))[0], 1)[0]
                            object_list_id, location_id = max_ind // rmap.shape[1], max_ind % rmap.shape[1]

                            location_token = location_id + locations_start
                            list_ids.append(object_list_id)
                            object_id = objects[object_list_id]
                            object_token = object_id + objects_start
                            rmap[:, location_id] = -np.inf
                            rmap[object_list_id, :] = -np.inf  
                        else:
                            object_id = N_OBJECTS
                            location_id = N_LOCATIONS
                            object_token = token_to_id['[LAST]']
                            location_token = token_to_id['[LAST]']

                        if step_i:
                            last_object_list_id = list_ids[step_i-1]
                            last_location_id = sequences[step_i-1][7] - locations_start
                            rack[last_location_id] = objects[last_object_list_id] + objects_start
                            # object_occupancy[last_object_list_id] = last_location_id + locations_start
                            object_occupancy[last_object_list_id] = token_to_id['[PLACED]']

                        step = torch.cat((step, torch.tensor(object_token).view(1,1)))
                        step = torch.cat((step, torch.tensor(location_token).view(1,1)))
                        step = torch.cat((step, copy.deepcopy(rack)))
                        step = torch.cat((step, copy.deepcopy(object_occupancy)))
                        step = step.view(1,-1)

                        sequences[t_seq_i * (len(objects)+1) + step_i] = step 

                pbar.update(1)

            if SAVE_DATA: 
                dataset_root = dataroot.joinpath(f'capacity_{config["CAPACITY"]}')
                dataset_root.mkdir(parents=True, exist_ok=True)
                dataset_name = dataset_root.joinpath(f'{datasets[dataset_i]}.pt')
                torch.save(sequences, dataset_name)

@hydra.main(version_base=None, config_path='configs', config_name='sample_sequences')
def main(config):
    sample_sequences(config)

if __name__ == '__main__':
    main()