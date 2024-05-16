# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import pickle
import time

import hydra
import numpy as np
import torch
torch.manual_seed(42)
import torch.nn as nn
import tqdm
import wandb

from earlystopping import EarlyStopping
from utils import model_classes

MAX_LOCATIONS = 100
MAX_OBJECTS = 100

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def experiment(config, dataset_pfx, dataset):
    log_to_wandb = config['log_to_wandb']
    model_class = model_classes[config.get('strategy_encoder')]
    device = config.get('device', 'cpu')
    K = config.get('K')
    bsz = config['batch_size']

    policy = config.get('policy')

    N_MEANS = config['N_MEANS']
    N_LOCATIONS = config['N_LOCATIONS']
    N_OBJECTS = config['N_OBJECTS']
    STDDEV = config['STDDEV']
    CAPACITY = config['CAPACITY']

    datasets_path = Path(f'{dataset_pfx}')
    dataset_path = datasets_path.joinpath(f'means_{N_MEANS}-locations_{N_LOCATIONS}-objects_{N_OBJECTS}-stddev_{STDDEV}')
    seqs_path = dataset_path.joinpath(f'capacity_{CAPACITY}')
    data_path = seqs_path.joinpath(f'{dataset}.pt')

    models_path = seqs_path.joinpath(f'models/{config["save_name"]}/{config["strategy_encoder"]}-{config["policy"]}-{config["K"]}')
    models_path.mkdir(parents=True, exist_ok=True)

    # load the dataset
    data = torch.load(data_path)
    d_rack = 100
    d_all = data.shape[1] 

    objective_path = dataset_path.joinpath(f'{dataset}_set.pt') 
    eval_objectives = torch.load(objective_path, map_location=torch.device('cuda'))
    # 1. load train split seqs
    # 2. load the eval split seqs
    last_ind = torch.sum(data[:,0] == data[:,0][0])-1
    test_data = data[data[:,5] != last_ind]

    weights_path = seqs_path.joinpath(f'train-ar_weights.pt')
    action_r_weights = torch.load(weights_path, map_location=torch.device('cuda')).to(device)

    loss_fn = [nn.CrossEntropyLoss(action_r_weights), nn.CrossEntropyLoss(reduction='sum')]

    def eval_prefs(model, params, objectives=eval_objectives):
        # Use for zero-shot evaluations
        all_accs = []
        all_regrets = []
        all_losses = []
        objs_repr = torch.eye(N_OBJECTS, dtype=torch.double)
        locs_repr = torch.eye(N_LOCATIONS, dtype=torch.double)

        eval_these = np.random.permutation(test_data[:,0].unique().numpy())

        acc_results = torch.zeros((eval_these.shape[0]))
        reg_results = torch.zeros((eval_these.shape[0]))

        with tqdm.tqdm(eval_these, desc=f'Eval Epoch', unit="trajectory") as t_eval_inds:
            state = torch.zeros((1,0,100), dtype=torch.long).to(device)
            prev_states = [torch.zeros((1,0,100), dtype=torch.long).to(device) for _ in range(K)]
            
            action_h = torch.zeros((1,0,1), dtype=torch.long).to(device)
            prev_actions_h = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]
            
            action_c = torch.zeros((1,0,1), dtype=torch.long).to(device)
            prev_actions_c = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

            action_r = torch.zeros((1,0,1), dtype=torch.long).to(device)

            timestep = torch.zeros((1,0,1), dtype=torch.long).to(device)
            
            for eid, ep_ind in enumerate(t_eval_inds):
                episode = test_data[test_data[:,0] == ep_ind].view(1,-1,d_all)
                objective_id = episode[0,0,3]
                objective = torch.tensor(objectives[objective_id])

                states = episode[:,:,8:108].clone().to(device)

                means = episode[:,:,2:3].clone().to(device)
                timesteps = episode[:,:,5:6].clone().to(device)
                actions_h = episode[:,:,6:7].clone().to(device)
                actions_r = episode[:,:,7:8].clone().to(device)
                actions_c = episode[:,:,7:8].clone().to(device)

                rmap = objs_repr @ objective.T @ locs_repr

                placement_map = torch.zeros_like(rmap)[0]

                min_reward = -2

                episode_regrets = []
                episode_losses = []
                for i in range(states.shape[1]-1):

                    for pid, pstate in enumerate(prev_states):
                        ind = max((i-(pid+1)), 0)
                        prev_states[pid] = torch.cat((prev_states[pid], states[0, ind].view(1,1,-1)), 1)
                        if prev_states[pid].shape[1] > K:
                            prev_states[pid] = prev_states[pid][:,1:]
                    
                    for aid, pah in enumerate(prev_actions_h):
                        ind = max((i-(aid+1)), 0)
                        prev_actions_h[aid] = torch.cat((prev_actions_h[aid], actions_h[0, ind].view(1,1,-1)), 1)
                        if prev_actions_h[pid].shape[1] > K:
                            prev_actions_h[pid] = prev_actions_h[pid][:,1:]

                    for aid, pac in enumerate(prev_actions_c):
                        ind = max((i-(aid+1)), 0)
                        prev_actions_c[aid] = torch.cat((prev_actions_c[aid], actions_c[0, ind].view(1,1,-1)), 1)
                        if prev_actions_c[pid].shape[1] > K:
                            prev_actions_c[pid] = prev_actions_c[pid][:,1:]

                    state = torch.cat((state, states[0,i].view(1,1,-1)), 1)
                    if state.shape[1] > K:
                        state = state[:,1:]
                    
                    action_h = torch.cat((action_h, actions_h[0,i].view(1,1,-1)), 1)
                    if action_h.shape[1] > K:
                        action_h = action_h[:,1:]

                    action_c = torch.cat((action_c, actions_c[0,i].view(1,1,-1)), 1)
                    if action_c.shape[1] > K:
                        action_c = action_c[:,1:]

                    tmp_action_c = action_c.clone()
                    tmp_action_r = action_r.clone()
                    tmp_action_r = torch.cat((tmp_action_r, torch.zeros((1,1,1)).to(device)), 1).to(torch.long)

                    timestep = torch.cat((timestep, timesteps[0,i].view(1,1,-1)), 1)
                    o = objective.to(device).to(torch.float) 

                    out = model.forward( # maybe change this to match better with outputs
                            state[:,-1:], 
                            action_h[:,-1:],
                            [prev_state[:,-1:] for prev_state in prev_states],
                            [prev_ah[:,-1:] for prev_ah in prev_actions_h],
                            [prev_ac[:,-1:] for prev_ac in prev_actions_c],
                            timestep[:,-1:], 
                            attention_mask=None, 
                            strategy=o,
                            c = means[:,i:i+1,:]
                        )


                    if config['policy'] == 'IRL':
                        pref_pred = out['theta_preds']
                        pred_probs = (objs_repr[action_h[:,-1].cpu()-objects_start] @ pref_pred[0,-1].view(N_LOCATIONS,N_OBJECTS).T.cpu().to(torch.double)) #+ placement_map
                        action_r_pred = torch.argmax(pred_probs + placement_map).to(device)

                        loss = loss_fn[1](
                            pred_probs.view(1,N_LOCATIONS).to(device), actions_c[:,i,:].view(-1) - locations_start
                        )
                        acc = torch.sum((action_r_pred == actions_c[:,i,:].view(-1))) / actions_c[:,i,:].view(-1).shape[0]
                    else:
                        pred_probs = out['action_r_preds'].reshape(-1,d_all)
                        action_r_pred = torch.argmax(pred_probs, -1) - locations_start
                
                        loss = loss_fn[0](
                            pred_probs, actions_c[:,i,:].view(-1)
                        )
                        acc = torch.sum((action_r_pred == actions_c[:,i,:].view(-1))) / actions_c[:,i,:].view(-1).shape[0]

                    action_r = torch.cat((action_r, action_r_pred.view(1,1,-1)), 1)
                    if action_r.shape[1] > K:
                        action_r = action_r[:,1:]

                    episode_losses.append(loss.item())

                    placement_map[action_c[0,-1].item()-locations_start] = -np.inf

                    # robot_reward = rmap[action_h[0,-1].item()-objects_start, action_r[0,-1].item()-locations_start].item() if action_r[0,-1].item()-locations_start < N_LOCATIONS else min_reward
                    robot_reward = rmap[action_h[0,-1].item()-objects_start, action_r[0,-1].item()].item() if action_r[0,-1].item() < N_LOCATIONS else min_reward
                    robot_reward = min_reward if np.isinf(robot_reward) else robot_reward

                    human_reward = rmap[action_h[0,-1].item()-objects_start, action_c[0,-1].item()-locations_start].item()

                    rmap[:, action_c[0,-1].item()-locations_start] = -np.inf

                    episode_regrets.append(human_reward - robot_reward)

                acc = (torch.sum(action_r[:,-N_LOCATIONS:] == (action_c[:,-N_LOCATIONS:]-locations_start)) / N_LOCATIONS).item()

                acc_results[eid] = acc
                reg_results[eid] = np.mean(episode_regrets)

                all_accs.append(acc)
                all_regrets.append(np.mean(episode_regrets))
                all_losses.append(np.mean(episode_losses))
                t_eval_inds.set_postfix(acc=np.mean(all_accs), loss=np.mean(all_losses), regret=np.mean(all_regrets))

        return {
            'accs': np.mean(all_accs),
            'regrets': np.mean(all_regrets),
            'losses': np.mean(all_losses),
            'accs_table_zeroshot': acc_results,
            'regs_table_zeroshot': reg_results
        }

    def eval_online(model, params, objectives=eval_objectives, eval_save_name=config['eval_save_name']):
        # Use for online evaluation of BLR-HAC and Linear models

        all_accs = []
        all_regrets = []
        all_losses = []
        objs_repr = torch.eye(N_OBJECTS, dtype=torch.double)
        locs_repr = torch.eye(N_LOCATIONS, dtype=torch.double)

        objective_ids = test_data[:,3].unique()#np.random.permutation()

        warm_start_counter = 1
        warm_start = 0
        reload_every = config['reload_every']
        num_eval = config['num_eval']

        acc_results = torch.zeros((objective_ids.shape[0], num_eval+warm_start_counter, N_OBJECTS))
        reg_results = torch.zeros((objective_ids.shape[0], num_eval+warm_start_counter, N_OBJECTS))

        pref_recon = torch.zeros((N_LOCATIONS, 1, N_OBJECTS*N_LOCATIONS)).to(device)
        tmp_updates = torch.zeros((N_OBJECTS, N_OBJECTS, N_LOCATIONS)).to(device)
        warm_start = 0
        model.load_state_dict(params)

        for oid, objective_id in enumerate(tqdm.tqdm(objective_ids, desc="Objectives", position=0)):
            episodes = test_data[test_data[:,3] == objective_id]
            eval_these = np.random.permutation(episodes[:,0].unique().numpy())[:num_eval + warm_start_counter]

            count = oid+1 if warm_start < warm_start_counter else oid+2
            if not count % reload_every:
                pref_recon = torch.zeros((N_LOCATIONS, 1, N_OBJECTS*N_LOCATIONS)).to(device)
                tmp_updates = torch.zeros((N_OBJECTS, N_OBJECTS, N_LOCATIONS)).to(device)
                warm_start = 0
                model.load_state_dict(params)

            model.eval()
            
            # warm_start = 0
            with tqdm.tqdm(eval_these, desc=f'Episodes', unit="trajectory", position=1, leave=False) as t_eval_inds:
                state = torch.zeros((1,0,100), dtype=torch.long).to(device)
                prev_states = [torch.zeros((1,0,100), dtype=torch.long).to(device) for _ in range(K)]

                action_h = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_h = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_c = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_c = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_r = torch.zeros((1,0,1), dtype=torch.long).to(device)

                timestep = torch.zeros((1,0,1), dtype=torch.long).to(device)

                # for each episode
                for eid, ep_ind in enumerate(t_eval_inds):
                    episode = test_data[test_data[:,0] == ep_ind].view(1,-1,d_all)
                    objective = torch.tensor(objectives[objective_id])

                    states = episode[:,:,8:108].clone().to(device)

                    means = episode[:,:,2:3].clone().to(device)
                    timesteps = episode[:,:,5:6].clone().to(device)
                    actions_h = episode[:,:,6:7].clone().to(device)
                    actions_r = episode[:,:,7:8].clone().to(device)
                    actions_c = episode[:,:,7:8].clone().to(device)

                    rmap = objs_repr @ objective.T @ locs_repr

                    placement_map = torch.zeros_like(rmap)[0]

                    min_reward = -2

                    episode_regrets = []
                    episode_losses = []

                    # for each step in the episode
                    for i in range(states.shape[1]):
                        for pid, pstate in enumerate(prev_states):
                            ind = max((i-(pid+1)), 0)
                            prev_states[pid] = torch.cat((prev_states[pid], states[0, ind].view(1,1,-1)), 1)
                            if prev_states[pid].shape[1] > K:
                                prev_states[pid] = prev_states[pid][:,1:]

                        for aid, pah in enumerate(prev_actions_h):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_h[aid] = torch.cat((prev_actions_h[aid], actions_h[0, ind].view(1,1,-1)), 1)
                            if prev_actions_h[pid].shape[1] > K:
                                prev_actions_h[pid] = prev_actions_h[pid][:,1:]

                        for aid, pac in enumerate(prev_actions_c):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_c[aid] = torch.cat((prev_actions_c[aid], actions_c[0, ind].view(1,1,-1)), 1)
                            if prev_actions_c[pid].shape[1] > K:
                                prev_actions_c[pid] = prev_actions_c[pid][:,1:]

                        state = torch.cat((state, states[0,i].view(1,1,-1)), 1)
                        if state.shape[1] > K:
                            state = state[:,1:]

                        action_h = torch.cat((action_h, actions_h[0,i].view(1,1,-1)), 1)
                        if action_h.shape[1] > K:
                            action_h = action_h[:,1:]

                        action_c = torch.cat((action_c, actions_c[0,i].view(1,1,-1)), 1)
                        if action_c.shape[1] > K:
                            action_c = action_c[:,1:]

                        tmp_action_c = action_c.clone()
                        tmp_action_r = action_r.clone()
                        tmp_action_r = torch.cat((tmp_action_r, torch.zeros((1,1,1)).to(device)), 1).to(torch.long)

                        timestep = torch.cat((timestep, timesteps[0,i].view(1,1,-1)), 1)
                        o = objective.to(device).to(torch.float) 

                        if warm_start < warm_start_counter and config['bootstrap']:
                            out = model.forward( # maybe change this to match better with outputs
                                    state[:,-1:], 
                                    action_h[:,-1:],
                                    [prev_state[:,-1:] for prev_state in prev_states],
                                    [prev_ah[:,-1:] for prev_ah in prev_actions_h],
                                    [prev_ac[:,-1:] for prev_ac in prev_actions_c],
                                    timestep[:,-1:], 
                                    attention_mask=None, 
                                    strategy=o,
                                    c = means[:,i:i+1,:]
                                )

                            pref_pred = out['theta_preds']
                            # pref_pred = torch.zeros_like(out['theta_preds'])
                            tmp = pref_pred[0]
                            pref_recon[i] = tmp #(pref_recon[i] * eid + tmp) / (eid+1) 
                        
                        pred_probs = (objs_repr[action_h[:,-1].cpu()-objects_start] @ pref_recon[i].view(N_LOCATIONS,N_OBJECTS).T.cpu().to(torch.double)) 

                        action_r_pred = torch.argmax(pred_probs + placement_map).to(device)+locations_start
                        action_r = torch.cat((action_r, action_r_pred.view(1,1,-1)), 1)
                        if action_r.shape[1] > K:
                            action_r = action_r[:,1:]

                        ah_features = objs_repr[action_h[:,-1].cpu()-objects_start]
                        ac_features = locs_repr[action_c[:,-1].cpu()-locations_start]
                        ar_features = locs_repr[action_r[:,-1].cpu()-locations_start]

                        update = -config['update_lr']*(ah_features[0,0].view(1,-1) * ac_features[0,0].view(-1,1) - ah_features[0,0].view(1,-1) * ar_features[0,0].view(-1,1)).to(device)
                        if warm_start < warm_start_counter:
                            tmp_updates[i] += update
                        else: 
                            pref_recon[i] -= update.view(1,-1)

                        loss = loss_fn[1](
                            pred_probs.view(1,N_LOCATIONS).to(device), actions_c[:,i,:].view(-1) - locations_start
                        )
                        episode_losses.append(loss.item())

                        placement_map[action_c[0,-1].item()-locations_start] = -np.inf

                        robot_reward = rmap[action_h[0,-1].item()-objects_start, action_r[0,-1].item()-locations_start].item() if action_r[0,-1].item()-locations_start < N_LOCATIONS else min_reward
                        robot_reward = min_reward if np.isinf(robot_reward) else robot_reward

                        human_reward = rmap[action_h[0,-1].item()-objects_start, action_c[0,-1].item()-locations_start].item()

                        rmap[:, action_c[0,-1].item()-locations_start] = -np.inf

                        episode_regrets.append(human_reward - robot_reward)
                        
                        acc_results[oid, eid, i] = action_c[0,-1].item() == action_r[0,-1].item()
                        reg_results[oid, eid, i] = human_reward - robot_reward

                    if warm_start < warm_start_counter:
                        if config['bootstrap']:
                            model.train()
                            for _ in range(5):
                                optimizer1.zero_grad()

                                out = model.forward( # maybe change this to match better with outputs
                                    state.view(state.shape[1],1,-1), 
                                    action_h.view(state.shape[1],1,-1), 
                                    [prev_state.view(state.shape[1],1,-1) for prev_state in prev_states],
                                    [prev_ah.view(state.shape[1],1,-1) for prev_ah in prev_actions_h],
                                    [prev_ac.view(state.shape[1],1,-1) for prev_ac in prev_actions_c],
                                    timestep, 
                                    attention_mask=None, 
                                    strategy=objectives,
                                    c = None
                                )

                                preds = out['theta_preds'].view(state.shape[1],N_LOCATIONS,N_OBJECTS)[torch.arange(state.shape[1]), :, (action_h-objects_start).view(-1)].view(state.shape[1], N_LOCATIONS)
                                loss = loss_fn[1](preds, action_c.view(-1) - locations_start)
                                loss.backward()
                                optimizer1.step()

                            model.eval()
                            out = model.forward( # maybe change this to match better with outputs
                                state.view(state.shape[1],1,-1), 
                                action_h.view(state.shape[1],1,-1), 
                                [prev_state.view(state.shape[1],1,-1) for prev_state in prev_states],
                                [prev_ah.view(state.shape[1],1,-1) for prev_ah in prev_actions_h],
                                [prev_ac.view(state.shape[1],1,-1) for prev_ac in prev_actions_c],
                                timestep, 
                                attention_mask=None, 
                                strategy=objectives,
                                c = None
                            )
                            pref_pred = out['theta_preds']
                            tmp = pref_pred.view(-1, N_LOCATIONS, N_OBJECTS*N_LOCATIONS).mean(0).unsqueeze(1)
                            pref_recon = tmp # (pref_recon[i] * num_updates + tmp) / (num_updates+1)

                        pref_recon -= tmp_updates.view(pref_recon.shape)
                    
                    warm_start+=1

                    acc = (torch.sum(action_r[:,-N_LOCATIONS:] == (action_c[:,-N_LOCATIONS:])) / N_LOCATIONS).item()

                    all_accs.append(acc)
                    all_regrets.append(np.mean(episode_regrets))
                    all_losses.append(np.mean(episode_losses))
                    t_eval_inds.set_postfix(acc=acc, loss=np.mean(episode_losses), regret=np.mean(episode_regrets))
                if not (oid+1) % reload_every:
                    print(acc_results[:oid+1,:eid+1].mean(-1).view((oid+1)//2,-1).mean(0)[1:-1])
                    # print(acc_results[:oid+1,:eid+1].mean(-1).view((oid+1),-1).mean(0)[1:])

        return {
            'accs': np.mean(all_accs),
            'regrets': np.mean(all_regrets),
            'losses': np.mean(all_losses),
            f'accs_table_{eval_save_name}': acc_results,
            f'regs_table_{eval_save_name}': reg_results
        }

    def eval_online_sgd(model, params, objectives=eval_objectives, eval_save_name=config['eval_save_name']):
        # Use for online evaluation of Transformer baseline

        all_accs = []
        all_regrets = []
        all_losses = []
        objs_repr = torch.eye(N_OBJECTS, dtype=torch.double)
        locs_repr = torch.eye(N_LOCATIONS, dtype=torch.double)

        objective_ids = np.random.permutation(test_data[:,3].unique())

        acc_results = torch.zeros((objective_ids.shape[0], 1000, N_OBJECTS))
        reg_results = torch.zeros((objective_ids.shape[0], 1000, N_OBJECTS))
        warm_start_counter = 1
        warm_start = 0
        update_counter = 1 # 1 is update every episode, large number is never update
        reload_every = config['reload_every']
        num_eval = config['num_eval']

        pref_recon = torch.zeros((N_LOCATIONS, 1, N_OBJECTS*N_LOCATIONS)).to(device)
        warm_start = 0
        model.load_state_dict(params)

        # for each objective
        for oid, objective_id in enumerate(tqdm.tqdm(objective_ids, desc="Objectives", position=0)):
            episodes = test_data[test_data[:,3] == objective_id]
            eval_these = np.random.permutation(episodes[:,0].unique().numpy())[:num_eval+warm_start_counter]

            count = oid + 1 if warm_start < warm_start_counter else oid + 2
            if not count % reload_every:
                pref_recon = torch.zeros((N_LOCATIONS, 1, N_OBJECTS*N_LOCATIONS)).to(device)
                warm_start = 0
                model.load_state_dict(params)

            model.eval()
            
            with tqdm.tqdm(eval_these, desc=f'Eval Epoch', unit="trajectory", position=1, leave=False) as t_eval_inds:
                state = torch.zeros((1,0,100), dtype=torch.long).to(device)
                prev_states = [torch.zeros((1,0,100), dtype=torch.long).to(device) for _ in range(K)]

                action_h = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_h = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_c = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_c = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_r = torch.zeros((1,0,1), dtype=torch.long).to(device)

                timestep = torch.zeros((1,0,1), dtype=torch.long).to(device)

                # for each episode
                for eid, ep_ind in enumerate(t_eval_inds):
                    episode = test_data[test_data[:,0] == ep_ind].view(1,-1,d_all)
                    # objective_id = episode[0,0,3]
                    objective = torch.tensor(objectives[objective_id])

                    states = episode[:,:,8:108].clone().to(device)

                    means = episode[:,:,2:3].clone().to(device)
                    timesteps = episode[:,:,5:6].clone().to(device)
                    actions_h = episode[:,:,6:7].clone().to(device)
                    actions_r = episode[:,:,7:8].clone().to(device)
                    actions_c = episode[:,:,7:8].clone().to(device)

                    rmap = objs_repr @ objective.T @ locs_repr

                    placement_map = torch.zeros_like(rmap)[0]

                    min_reward = -1

                    episode_regrets = []
                    episode_losses = []

                    # for each step
                    for i in range(states.shape[1]):
                        for pid, pstate in enumerate(prev_states):
                            ind = max((i-(pid+1)), 0)
                            prev_states[pid] = torch.cat((prev_states[pid], states[0, ind].view(1,1,-1)), 1)
                            if prev_states[pid].shape[1] > K:
                                prev_states[pid] = prev_states[pid][:,1:]

                        for aid, pah in enumerate(prev_actions_h):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_h[aid] = torch.cat((prev_actions_h[aid], actions_h[0, ind].view(1,1,-1)), 1)
                            if prev_actions_h[aid].shape[1] > K:
                                prev_actions_h[aid] = prev_actions_h[aid][:,1:]

                        for aid, pac in enumerate(prev_actions_c):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_c[aid] = torch.cat((prev_actions_c[aid], actions_c[0, ind].view(1,1,-1)), 1)
                            if prev_actions_c[aid].shape[1] > K:
                                prev_actions_c[aid] = prev_actions_c[aid][:,1:]

                        state = torch.cat((state, states[0,i].view(1,1,-1)), 1)
                        if state.shape[1] > K:
                            state = state[:,1:]

                        action_h = torch.cat((action_h, actions_h[0,i].view(1,1,-1)), 1)
                        if action_h.shape[1] > K:
                            action_h = action_h[:,1:]

                        action_c = torch.cat((action_c, actions_c[0,i].view(1,1,-1)), 1)
                        if action_c.shape[1] > K:
                            action_c = action_c[:,1:]

                        tmp_action_c = action_c.clone()
                        tmp_action_r = action_r.clone()
                        tmp_action_r = torch.cat((tmp_action_r, torch.zeros((1,1,1)).to(device)), 1).to(torch.long)

                        timestep = torch.cat((timestep, timesteps[0,i].view(1,1,-1)), 1)
                        if timestep.shape[1] > K:
                            timestep = timestep[:,timestep.shape[1] - K:]
                        o = objective.to(device).to(torch.float) 

                        if warm_start < warm_start_counter:
                            out = model.forward( # maybe change this to match better with outputs
                                    state[:,-1:], 
                                    action_h[:,-1:],
                                    [prev_state[:,-1:] for prev_state in prev_states],
                                    [prev_ah[:,-1:] for prev_ah in prev_actions_h],
                                    [prev_ac[:,-1:] for prev_ac in prev_actions_c],
                                    timestep[:,-1:], 
                                    attention_mask=None, 
                                    strategy=o,
                                    c = means[:,i:i+1,:]
                                )

                            pref_pred = out['theta_preds']
                            tmp = pref_pred[0]
                            pref_recon[i] = pref_pred[0] 

                            # resample every step of warm start
                            # update after every episode of warm start
                            # use the final preference after this period until end of objective episode
                            #    resample + retrain after every M episodes

                        pred_probs = (objs_repr[action_h[:,-1].cpu()-objects_start] @ pref_recon[i].view(N_LOCATIONS,N_OBJECTS).T.cpu().to(torch.double)) 

                        action_r_pred = torch.argmax(pred_probs + placement_map).to(device)+locations_start
                        action_r = torch.cat((action_r, action_r_pred.view(1,1,-1)), 1)
                        if action_r.shape[1] > K:
                            action_r = action_r[:,1:]

                        loss = loss_fn[1](
                            pred_probs.view(1,N_LOCATIONS).to(device), actions_c[:,i,:].view(-1) - locations_start
                        )
                        episode_losses.append(loss.item())

                        placement_map[action_c[0,-1].item()-locations_start] = -np.inf

                        robot_reward = rmap[action_h[0,-1].item()-objects_start, action_r[0,-1].item()-locations_start].item() if action_r[0,-1].item()-locations_start < N_LOCATIONS else min_reward
                        robot_reward = min_reward if np.isinf(robot_reward) else robot_reward

                        human_reward = rmap[action_h[0,-1].item()-objects_start, action_c[0,-1].item()-locations_start].item()

                        rmap[:, action_c[0,-1].item()-locations_start] = -np.inf

                        acc_results[oid, eid, i] = action_c[0,-1].item() == action_r[0,-1].item()
                        reg_results[oid, eid, i] = human_reward - robot_reward

                    if warm_start < warm_start_counter or ((eid % update_counter) == 0):
                        model.train()
                        for _ in range(5):
                            optimizer1.zero_grad()

                            out = model.forward( # maybe change this to match better with outputs
                                state.view(state.shape[1],1,-1), 
                                action_h.view(state.shape[1],1,-1), 
                                [prev_state.view(state.shape[1],1,-1) for prev_state in prev_states],
                                [prev_ah.view(state.shape[1],1,-1) for prev_ah in prev_actions_h],
                                [prev_ac.view(state.shape[1],1,-1) for prev_ac in prev_actions_c],
                                timestep, 
                                attention_mask=None, 
                                strategy=objectives,
                                c = None
                            )

                            preds = out['theta_preds'].view(state.shape[1],N_LOCATIONS,N_OBJECTS)[torch.arange(state.shape[1]), :, (action_h-objects_start).view(-1)].view(state.shape[1], N_LOCATIONS)
                            loss = loss_fn[1](preds, action_c.view(-1) - locations_start)
                            loss.backward()
                            optimizer1.step()

                        model.eval()
                        out = model.forward( # maybe change this to match better with outputs
                            state.view(state.shape[1],1,-1), 
                            action_h.view(state.shape[1],1,-1), 
                            [prev_state.view(state.shape[1],1,-1) for prev_state in prev_states],
                            [prev_ah.view(state.shape[1],1,-1) for prev_ah in prev_actions_h],
                            [prev_ac.view(state.shape[1],1,-1) for prev_ac in prev_actions_c],
                            timestep, 
                            attention_mask=None, 
                            strategy=objectives,
                            c = None
                        )
                        pref_pred = out['theta_preds']
                        tmp = pref_pred.view(-1, N_LOCATIONS, N_OBJECTS*N_LOCATIONS).mean(0).unsqueeze(1)
                        pref_recon = tmp # (pref_recon[i] * num_updates + tmp) / (num_updates+1) 

                    warm_start+=1

                    acc = (torch.sum(action_r[:,-N_LOCATIONS:] == (action_c[:,-N_LOCATIONS:])) / N_LOCATIONS).item()

                    all_accs.append(acc)
                    all_regrets.append(np.mean(episode_regrets))
                    all_losses.append(np.mean(episode_losses))
                    t_eval_inds.set_postfix(acc=np.mean(all_accs), loss=np.mean(all_losses), regret=np.mean(all_regrets))
                if not (oid+1) % reload_every:
                    print(acc_results[:oid+1,:eid+1].mean(-1).view(-1, (num_eval+warm_start_counter)*reload_every).mean(0))
        return {
            'accs': np.mean(all_accs),
            'regrets': np.mean(all_regrets),
            'losses': np.mean(all_losses),
            f'accs_table_{eval_save_name}': acc_results,
            f'regs_table_{eval_save_name}': reg_results
        }


    def eval_online_switching(model, params, objectives=eval_objectives):
        # Use for nonstationary online evaluation of BLR-HAC and Linear models
        all_accs = []
        all_regrets = []
        all_losses = []
        objs_repr = torch.eye(N_OBJECTS, dtype=torch.double)
        locs_repr = torch.eye(N_LOCATIONS, dtype=torch.double)

        objective_ids = np.random.permutation(test_data[:,3].unique())
        acc_results = torch.zeros((objective_ids.shape[0], 250))
        reg_results = torch.zeros((objective_ids.shape[0], 250))

        warm_start = True

        for oid, objective_id in enumerate(tqdm.tqdm(objective_ids, desc="Objectives", position=0)):
            episodes = test_data[test_data[:,3] == objective_id]
            eval_these = np.random.permutation(episodes[:,0].unique().numpy())[:250]

            if not oid % 4:
                pref_recon = torch.zeros((1, 1, N_OBJECTS*N_LOCATIONS)).to(device)
                warm_start = True

            with tqdm.tqdm(eval_these, desc=f'Eval Epoch', unit="trajectory") as t_eval_inds:
                state = torch.zeros((1,0,100), dtype=torch.long).to(device)
                prev_states = [torch.zeros((1,0,100), dtype=torch.long).to(device) for _ in range(K)]

                action_h = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_h = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_c = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_c = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_r = torch.zeros((1,0,1), dtype=torch.long).to(device)

                timestep = torch.zeros((1,0,1), dtype=torch.long).to(device)

                # for each episode
                for eid, ep_ind in enumerate(t_eval_inds):
                    episode = test_data[test_data[:,0] == ep_ind].view(1,-1,d_all)
                    objective = torch.tensor(objectives[objective_id])

                    states = episode[:,:,8:108].clone().to(device)

                    means = episode[:,:,2:3].clone().to(device)
                    timesteps = episode[:,:,5:6].clone().to(device)
                    actions_h = episode[:,:,6:7].clone().to(device)
                    actions_r = episode[:,:,7:8].clone().to(device)
                    actions_c = episode[:,:,7:8].clone().to(device)

                    rmap = objs_repr @ objective.T @ locs_repr

                    placement_map = torch.zeros_like(rmap)[0]

                    min_reward = -1

                    episode_regrets = []
                    episode_losses = []

                    # for each step
                    for i in range(states.shape[1]-1):
                        for pid, pstate in enumerate(prev_states):
                            ind = max((i-(pid+1)), 0)
                            prev_states[pid] = torch.cat((prev_states[pid], states[0, ind].view(1,1,-1)), 1)
                            if prev_states[pid].shape[1] > K:
                                prev_states[pid] = prev_states[pid][:,1:]

                        for aid, pah in enumerate(prev_actions_h):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_h[aid] = torch.cat((prev_actions_h[aid], actions_h[0, ind].view(1,1,-1)), 1)
                            if prev_actions_h[aid].shape[1] > K:
                                prev_actions_h[aid] = prev_actions_h[aid][:,1:]

                        for aid, pac in enumerate(prev_actions_c):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_c[aid] = torch.cat((prev_actions_c[aid], actions_c[0, ind].view(1,1,-1)), 1)
                            if prev_actions_c[aid].shape[1] > K:
                                prev_actions_c[aid] = prev_actions_c[aid][:,1:]

                        state = torch.cat((state, states[0,i].view(1,1,-1)), 1)
                        if state.shape[1] > K:
                            state = state[:,1:]

                        action_h = torch.cat((action_h, actions_h[0,i].view(1,1,-1)), 1)
                        if action_h.shape[1] > K:
                            action_h = action_h[:,1:]

                        action_c = torch.cat((action_c, actions_c[0,i].view(1,1,-1)), 1)
                        if action_c.shape[1] > K:
                            action_c = action_c[:,1:]

                        tmp_action_c = action_c.clone()
                        tmp_action_r = action_r.clone()
                        tmp_action_r = torch.cat((tmp_action_r, torch.zeros((1,1,1)).to(device)), 1).to(torch.long)

                        timestep = torch.cat((timestep, timesteps[0,i].view(1,1,-1)), 1)
                        if timestep.shape[1] > K:
                            timestep = timestep[:,timestep.shape[1] - K:]
                        o = objective.to(device).to(torch.float) 

                        out = model.forward( # maybe change this to match better with outputs
                                state[:,-1:], 
                                action_h[:,-1:],
                                [prev_state[:,-1:] for prev_state in prev_states],
                                [prev_ah[:,-1:] for prev_ah in prev_actions_h],
                                [prev_ac[:,-1:] for prev_ac in prev_actions_c],
                                timestep[:,-1:], 
                                attention_mask=None, 
                                strategy=o,
                                c = means[:,i:i+1,:]
                            )

                        pred_probs = out['action_r_preds'].reshape(-1,d_all)
                        preds = torch.argmax(pred_probs, -1)
                        loss = loss_fn[0](
                            pred_probs, actions_c[:,i,:].view(-1)
                        )
                        acc = torch.sum((preds == actions_c[:,i,:].view(-1))) / actions_c[:,i,:].view(-1).shape[0]

                        pref_pred = out['theta_preds']

                        if warm_start:
                            pref_recon += pref_pred
                            pred_probs = (objs_repr[action_h[:,-1].cpu()-objects_start] @ pref_pred[0,-1].view(N_LOCATIONS,N_OBJECTS).T.cpu().to(torch.double)) 
                        else:
                            pred_probs = (objs_repr[action_h[:,-1].cpu()-objects_start] @ pref_recon[0,-1].view(N_LOCATIONS,N_OBJECTS).T.cpu().to(torch.double)) 

                        action_r_pred = torch.argmax(pred_probs + placement_map).to(device)+locations_start
                        action_r = torch.cat((action_r, action_r_pred.view(1,1,-1)), 1)
                        if action_r.shape[1] > K:
                            action_r = action_r[:,1:]

                        ah_features = objs_repr[action_h[:,-1].cpu()-objects_start]
                        ac_features = locs_repr[action_c[:,-1].cpu()-locations_start]
                        ar_features = locs_repr[action_r[:,-1].cpu()-locations_start]

                        update = -(ah_features[0,0].view(1,-1) * ac_features[0,0].view(-1,1) - ah_features[0,0].view(1,-1) * ar_features[0,0].view(-1,1)).to(device)
                        if not warm_start:
                            pref_recon -= update.view(1,1,-1)

                        loss = loss_fn[1](
                            pred_probs.view(1,N_LOCATIONS).to(device), actions_c[:,i,:].view(-1) - locations_start
                        )
                        episode_losses.append(loss.item())

                        placement_map[action_c[0,-1].item()-locations_start] = -np.inf

                        robot_reward = rmap[action_h[0,-1].item()-objects_start, action_r[0,-1].item()-locations_start].item() if action_r[0,-1].item()-locations_start < N_LOCATIONS else min_reward
                        robot_reward = min_reward if np.isinf(robot_reward) else robot_reward

                        human_reward = rmap[action_h[0,-1].item()-objects_start, action_c[0,-1].item()-locations_start].item()

                        rmap[:, action_c[0,-1].item()-locations_start] = -np.inf

                        episode_regrets.append(human_reward - robot_reward)

                    if warm_start:
                        pref_recon /= i
                    warm_start = False

                    acc = (torch.sum(action_r[:,-N_LOCATIONS:] == (action_c[:,-N_LOCATIONS:])) / N_LOCATIONS).item()

                    acc_results[objective_id, eid] = acc
                    reg_results[objective_id, eid] = np.mean(episode_regrets)

                    all_accs.append(acc)
                    all_regrets.append(np.mean(episode_regrets))
                    all_losses.append(np.mean(episode_losses))
                    t_eval_inds.set_postfix(acc=acc, loss=np.mean(episode_losses), regret=np.mean(episode_regrets))

        return {
            'accs': np.mean(all_accs),
            'regrets': np.mean(all_regrets),
            'losses': np.mean(all_losses),
            'accs_table_switching': acc_results,
            'regs_table_switching': reg_results
        }

    def eval_switching_sgd(model, params, objectives=eval_objectives):
        # Use for nonstationary online evaluation of Transformer baseline

        all_accs = []
        all_regrets = []
        all_losses = []
        objs_repr = torch.eye(N_OBJECTS, dtype=torch.double)
        locs_repr = torch.eye(N_LOCATIONS, dtype=torch.double)

        objective_ids = np.random.permutation(test_data[:,3].unique())

        num_eval = 9
        warm_start_counter = 1
        switch_every = 2

        acc_results = torch.zeros((objective_ids.shape[0], num_eval+warm_start_counter, N_OBJECTS))
        reg_results = torch.zeros((objective_ids.shape[0], num_eval+warm_start_counter, N_OBJECTS))
        update_counter = 10000

        # for each objective
        for oid, objective_id in enumerate(tqdm.tqdm(objective_ids, desc="Objectives", position=0)):
            episodes = test_data[test_data[:,3] == objective_id]
            eval_these = np.random.permutation(episodes[:,0].unique().numpy())[:num_eval+warm_start_counter]

            if not oid % switch_every:
                pref_recon = torch.zeros((N_LOCATIONS, 1, N_OBJECTS*N_LOCATIONS)).to(device)
                warm_start = 0
                model.load_state_dict(params)
                model.eval()

            with tqdm.tqdm(eval_these, desc=f'Eval Epoch', unit="trajectory", position=1, leave=False) as t_eval_inds:
                state = torch.zeros((1,0,100), dtype=torch.long).to(device)
                prev_states = [torch.zeros((1,0,100), dtype=torch.long).to(device) for _ in range(K)]

                action_h = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_h = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_c = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_c = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_r = torch.zeros((1,0,1), dtype=torch.long).to(device)

                timestep = torch.zeros((1,0,1), dtype=torch.long).to(device)

                # for each episode
                for eid, ep_ind in enumerate(t_eval_inds):
                    episode = test_data[test_data[:,0] == ep_ind].view(1,-1,d_all)
                    # objective_id = episode[0,0,3]
                    objective = torch.tensor(objectives[objective_id])

                    states = episode[:,:,8:108].clone().to(device)

                    means = episode[:,:,2:3].clone().to(device)
                    timesteps = episode[:,:,5:6].clone().to(device)
                    actions_h = episode[:,:,6:7].clone().to(device)
                    actions_r = episode[:,:,7:8].clone().to(device)
                    actions_c = episode[:,:,7:8].clone().to(device)

                    rmap = objs_repr @ objective.T @ locs_repr

                    placement_map = torch.zeros_like(rmap)[0]

                    min_reward = -1

                    episode_regrets = []
                    episode_losses = []

                    # for each step
                    for i in range(states.shape[1]):
                        for pid, pstate in enumerate(prev_states):
                            ind = max((i-(pid+1)), 0)
                            prev_states[pid] = torch.cat((prev_states[pid], states[0, ind].view(1,1,-1)), 1)
                            if prev_states[pid].shape[1] > K:
                                prev_states[pid] = prev_states[pid][:,1:]

                        for aid, pah in enumerate(prev_actions_h):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_h[aid] = torch.cat((prev_actions_h[aid], actions_h[0, ind].view(1,1,-1)), 1)
                            if prev_actions_h[aid].shape[1] > K:
                                prev_actions_h[aid] = prev_actions_h[aid][:,1:]

                        for aid, pac in enumerate(prev_actions_c):
                            ind = max((i-(aid+1)), 0)
                            prev_actions_c[aid] = torch.cat((prev_actions_c[aid], actions_c[0, ind].view(1,1,-1)), 1)
                            if prev_actions_c[aid].shape[1] > K:
                                prev_actions_c[aid] = prev_actions_c[aid][:,1:]

                        state = torch.cat((state, states[0,i].view(1,1,-1)), 1)
                        if state.shape[1] > K:
                            state = state[:,1:]

                        action_h = torch.cat((action_h, actions_h[0,i].view(1,1,-1)), 1)
                        if action_h.shape[1] > K:
                            action_h = action_h[:,1:]

                        action_c = torch.cat((action_c, actions_c[0,i].view(1,1,-1)), 1)
                        if action_c.shape[1] > K:
                            action_c = action_c[:,1:]

                        tmp_action_c = action_c.clone()
                        tmp_action_r = action_r.clone()
                        tmp_action_r = torch.cat((tmp_action_r, torch.zeros((1,1,1)).to(device)), 1).to(torch.long)

                        timestep = torch.cat((timestep, timesteps[0,i].view(1,1,-1)), 1)
                        if timestep.shape[1] > K:
                            timestep = timestep[:,timestep.shape[1] - K:]
                        o = objective.to(device).to(torch.float) 

                        if warm_start < warm_start_counter:
                            out = model.forward( # maybe change this to match better with outputs
                                    state[:,-1:], 
                                    action_h[:,-1:],
                                    [prev_state[:,-1:] for prev_state in prev_states],
                                    [prev_ah[:,-1:] for prev_ah in prev_actions_h],
                                    [prev_ac[:,-1:] for prev_ac in prev_actions_c],
                                    timestep[:,-1:], 
                                    attention_mask=None, 
                                    strategy=o,
                                    c = means[:,i:i+1,:]
                                )

                            pref_pred = out['theta_preds']
                            tmp = pref_pred[0]
                            pref_recon[i] = pref_pred[0] # (pref_recon[i] * num_updates[i] + tmp) / (num_updates[i]+1) 
                            # num_updates[i] += 1

                            # resample every step of warm start
                            # update after every episode of warm start
                            # use the final preference after this period until end of objective episode
                            #    resample + retrain after every M episodes

                        pred_probs = (objs_repr[action_h[:,-1].cpu()-objects_start] @ pref_recon[i].view(N_LOCATIONS,N_OBJECTS).T.cpu().to(torch.double)) 

                        action_r_pred = torch.argmax(pred_probs + placement_map).to(device)+locations_start
                        action_r = torch.cat((action_r, action_r_pred.view(1,1,-1)), 1)
                        if action_r.shape[1] > K:
                            action_r = action_r[:,1:]

                        loss = loss_fn[1](
                            pred_probs.view(1,N_LOCATIONS).to(device), actions_c[:,i,:].view(-1) - locations_start
                        )
                        episode_losses.append(loss.item())

                        placement_map[action_c[0,-1].item()-locations_start] = -np.inf

                        robot_reward = rmap[action_h[0,-1].item()-objects_start, action_r[0,-1].item()-locations_start].item() if action_r[0,-1].item()-locations_start < N_LOCATIONS else min_reward
                        robot_reward = min_reward if np.isinf(robot_reward) else robot_reward

                        human_reward = rmap[action_h[0,-1].item()-objects_start, action_c[0,-1].item()-locations_start].item()

                        rmap[:, action_c[0,-1].item()-locations_start] = -np.inf

                        # episode_regrets.append()

                        acc_results[oid, eid, i] = action_c[0,-1].item() == action_r[0,-1].item()
                        reg_results[oid, eid, i] = human_reward - robot_reward

                    if warm_start < warm_start_counter or ((eid % update_counter) == 0):
                        model.train()
                        for _ in range(5):
                            optimizer1.zero_grad()

                            out = model.forward( # maybe change this to match better with outputs
                                state.view(state.shape[1],1,-1), 
                                action_h.view(state.shape[1],1,-1), 
                                [prev_state.view(state.shape[1],1,-1) for prev_state in prev_states],
                                [prev_ah.view(state.shape[1],1,-1) for prev_ah in prev_actions_h],
                                [prev_ac.view(state.shape[1],1,-1) for prev_ac in prev_actions_c],
                                timestep, 
                                attention_mask=None, 
                                strategy=objectives,
                                c = None
                            )

                            preds = out['theta_preds'].view(state.shape[1],N_LOCATIONS,N_OBJECTS)[torch.arange(state.shape[1]), :, (action_h-objects_start).view(-1)].view(state.shape[1], N_LOCATIONS)
                            loss = loss_fn[1](preds, action_c.view(-1) - locations_start)
                            loss.backward()
                            optimizer1.step()

                        model.eval()
                        out = model.forward( # maybe change this to match better with outputs
                            state.view(state.shape[1],1,-1), 
                            action_h.view(state.shape[1],1,-1), 
                            [prev_state.view(state.shape[1],1,-1) for prev_state in prev_states],
                            [prev_ah.view(state.shape[1],1,-1) for prev_ah in prev_actions_h],
                            [prev_ac.view(state.shape[1],1,-1) for prev_ac in prev_actions_c],
                            timestep, 
                            attention_mask=None, 
                            strategy=objectives,
                            c = None
                        )
                        pref_pred = out['theta_preds']
                        # tmp = pref_pred[-N_LOCATIONS:]
                        tmp = pref_pred.view(-1, N_LOCATIONS, N_OBJECTS*N_LOCATIONS).mean(0).unsqueeze(1)
                        pref_recon = tmp # (pref_recon[i] * num_updates + tmp) / (num_updates+1) 

                    warm_start+=1

                    acc = (torch.sum(action_r[:,-N_LOCATIONS:] == (action_c[:,-N_LOCATIONS:])) / N_LOCATIONS).item()

                    # acc_results[oid, eid] = acc
                    # reg_results[oid, eid] = np.mean(episode_regrets)

                    all_accs.append(acc)
                    all_regrets.append(np.mean(episode_regrets))
                    all_losses.append(np.mean(episode_losses))
                    t_eval_inds.set_postfix(acc=np.mean(all_accs), loss=np.mean(all_losses), regret=np.mean(all_regrets))
                if not (oid+1) % 4:
                    print(acc_results[:oid+1].mean(-1).view(-1, (num_eval+warm_start_counter)*switch_every).mean(0))

        return {
            'accs': np.mean(all_accs),
            'regrets': np.mean(all_regrets),
            'losses': np.mean(all_losses),
            'accs_table_adaptation_sgd': acc_results,
            'regs_table_adaptation_sgd': reg_results
        }


    activation_function = config.get('activation_function') if config['strategy_encoder'] in ['MLP', 'Transformer'] else None

    model = model_class(
        sz_vocab = config.get('sz_vocab'),#, 13),
        d_embed = config.get('d_embed'),#, 128),
        max_length = config.get('K'),#, 20), # number of steps to keep in context
        max_ep_len = config.get('max_ep_len'),#, 7), # max length of each episode, equivalent to N_LOCATIONS * CAPACITY
        n_act_layers = config.get('n_act_layers'), #, 3), # need to rename to n_layer when passing to transformer
        pdrop = config.get('dropout'), #, .1),
        n_objects = N_OBJECTS,
        n_locations = N_LOCATIONS,
        sz_state = d_rack,
        K = config.get('K'),
        n_positions = config.get('n_positions'),
        d_hidden = config.get('d_hidden'),
        n_pref_layers = config.get('n_pref_layers'),
        n_head = config.get('n_head'),
        n_inner = None if not config.get('n_inner') else config.get('n_inner'),
        activation_function = activation_function,
        resid_pdrop = config.get('resid_pdrop'),
        embd_pdrop = config.get('embd_pdrop'),
        attn_pdrop = config.get('attn_pdrop'),
    ).to(device)

    print(count_parameters(model))
    params = None
    if config['load_from_state']:
        params = torch.load(f'{models_path}/{config["state_name"]}.pt', map_location=torch.device('cuda'))
        model.load_state_dict(params)

    optimizer1 = torch.optim.SGD(
        list(model.parameters()),
        lr=config['learning_rate'],
    )

    optimizer2 = torch.optim.SGD(
        list(model.pred_action.parameters()),
        lr=config['learning_rate'],
    )

    all_eval_fns = {
        'eval_prefs': eval_prefs,
        'eval_online': eval_online,
        'eval_online_sgd': eval_online_sgd,
        'eval_online_switching': eval_online_switching,
        'eval_switching_sgd': eval_switching_sgd,
    }

    scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, factor=0.5)
    trainer = Trainer(
        model=model,
        optimizer=[optimizer1, optimizer2],
        batch_size=bsz,
        get_batch = None, #get_batch_act2, #get_batch_act if act else get_batch,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=[all_eval_fns[config['eval_fn']]],
        k=config['K'],
        d_all=d_all,
        device=device,
        n_objects=N_OBJECTS,
        n_locations=N_LOCATIONS,
        policy=policy,
        params = params
    )

    if log_to_wandb:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project=config["wandb_name"],
            # Track hyperparameters and run metadata
            config=config
        )

    outputs = trainer.train_iteration(num_steps=config['num_steps_per_iter'], iter_num=1, print_logs=True)
    if log_to_wandb:
        wandb.log(outputs['logs'])
    
    with open(models_path.joinpath('logs.pkl'), 'wb') as f:
        pickle.dump(outputs['logs'], f)

    tables = outputs['tables']
    if Path.is_file(models_path.joinpath('tables.pt')):
        tables = torch.load(models_path.joinpath('tables.pt'), map_location=torch.device('cuda'))
        for k,v in outputs['tables'].items():
            tables[k] = v
    torch.save(tables, models_path.joinpath('tables.pt'))

class Trainer:
    ### grabbed from: https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/decision_transformer/training/trainer.py
    ### MIT LICENSE 
    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, early_stopping=None, k=None, d_all=None, n_objects=None, n_locations=None, policy='IRL', device='cpu', params=None):
        self.model = model
        self.optimizers = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.d_all = d_all
        self.device = device
        self.sm = nn.Softmax(-1).to(self.device)
        self.n_objects = n_objects
        self.n_locations = n_locations
        self.policy = policy
        self.params = params

        self.start_time = time.time()
        self.k = k

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        logs = dict()
        tables = dict()

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model, self.params)
            for k, v in outputs.items():
                if type(v) is not torch.Tensor:
                    logs[f'evaluation/{k}'] = v
                else:
                    tables[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return {
            'logs': logs, 
            'tables': tables
        }

@hydra.main(version_base=None, config_path='configs', config_name='sweep-train-dt-0')
def main(config):
    dataset_pfx = 'datasets'
    dataset = 'test_close'
    experiment(config, dataset_pfx, dataset)

if __name__ == '__main__':
    main()


"""
for each objective
    pref = torch.zeros_like(pref)
    for each episode
        for each step
            gather inputs
            out = model.forward
            pred = out['preds']

"""