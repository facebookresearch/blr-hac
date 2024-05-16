# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import pickle
import time

import hydra
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
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
    
    num_eval_episodes = config['num_eval_episodes']

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
    d_extra = 6
    d_rack = 100
    d_objects = 100
    d_ah = 1
    d_ar = 1
    d_all = data.shape[1] #sum([d_extra,d_state,d_ah,d_ar,d_ac])

    objective_path = dataset_path.joinpath(f'{dataset}_set.pt') #f'{dataset_pfx}/{dataset}-objectives.pt'
    train_objectives = torch.load(objective_path)

    split_path = seqs_path.joinpath(f'{dataset}-train_split.pt')
    if not split_path.is_file():
        seq_ids = np.unique(data[:,0])
        sz_eval_objectives = int(np.random.permutation(torch.unique(data[:,3])).shape[0] * .1)
        eval_objectives = np.random.permutation(torch.unique(data[:,3]))[:sz_eval_objectives]
        torch.save(eval_objectives, seqs_path.joinpath('eval_objectives.pt'))

        train_data = []
        eval_data = []
        for id in seq_ids:
            traj = data[data[:,0] == id]
            if traj[0,3].item() in eval_objectives:
                eval_data.append(traj)
            else:
                train_data.append(traj)
        train_data = torch.cat(train_data)
        eval_data = torch.cat(eval_data)

        torch.save(train_data, seqs_path.joinpath(f'{dataset}-train_split.pt'))
        torch.save(eval_data, seqs_path.joinpath(f'{dataset}-eval_split.pt'))

        # 1. generate train split objectives
        # 2. generate eval split objectives
        # 3. generate train split seqs
        # 4. generate eval split seqs
        # 5. save all of these
    else:
        train_data = torch.load(seqs_path.joinpath(f'{dataset}-train_split.pt'))
        eval_data = torch.load(seqs_path.joinpath(f'{dataset}-eval_split.pt'))

        # 1. load train split seqs
        # 2. load the eval split seqs
    last_ind = torch.sum(train_data[:,0] == train_data[:,0][0])-1
    train_data = train_data[train_data[:,5] != last_ind]
    eval_data = eval_data[eval_data[:,5] != last_ind]
    
    weights_path = seqs_path.joinpath(f'{dataset}-ar_weights.pt')
    if not weights_path.is_file():
        C = sorted(train_data[:,7].unique().tolist())
        a = compute_class_weight('balanced', classes=C, y=train_data[:,7].contiguous().view(-1).tolist())
        weights = torch.ones(d_all)*.01
        for i,c in enumerate(C):
            weights[c] = a[i]
        weights = torch.save(weights, weights_path)
    action_r_weights = torch.load(weights_path).to(device)
    
    loss_fn = [nn.CrossEntropyLoss(action_r_weights), nn.CrossEntropyLoss(reduction='sum')]

    def get_batch(bsz, K=K, dact=False, step=1, objectives=train_objectives):
        # return the current state, current human action, previous state, previous human action, previous correction

        batch_inds = np.random.choice(np.arange(train_data.shape[0]), bsz, replace=True)
        batch_step_ids = train_data[batch_inds, 5]

        batches_inds = [batch_inds]
        for i in range(1,K+1):
            step_ids = batch_step_ids - i
            inds = batch_inds - i
            inds[step_ids < 0] = batches_inds[-1][step_ids < 0]
            batches_inds.append(inds)

        future_batches_inds = [batch_inds]
        masks = []
        for i in range(N_LOCATIONS-1):
            step_ids = batch_step_ids + i + 1
            inds = batch_inds + i + 1
            inds[step_ids >= N_LOCATIONS] = future_batches_inds[-1][step_ids >= N_LOCATIONS]
            mask = np.zeros_like(inds)
            mask[step_ids >= N_LOCATIONS] = 1
            future_batches_inds.append(inds)
            masks.append(mask)

        batches = [train_data[inds].clone().view(inds.shape[0], 1, d_all) for inds in batches_inds]
        future_batches = [train_data[inds].clone().view(inds.shape[0], 1, d_all) for inds in future_batches_inds]
        thetas = [torch.tensor(objectives[batch[:,:,3]]).to(torch.float).to(device) for batch in batches]

        def get_data(ebatch, device, dact, masks=None): 
            if masks is None:
                masks = torch.ones((ebatch.shape[0], 1), dtype=torch.long).to(device)
            else:
                masks = torch.tensor(masks)

            timesteps = ebatch[:,:,5:6].clone().to(device)
            actions_h = ebatch[:,:,6:7].clone().to(device)
            actions_r = ebatch[:,:,7:8].clone()
            if dact: # random choice for some actions
                actions_inds = np.arange(actions_r.view(-1).shape[0])[masks.view(-1).cpu().numpy() == 1]
                actions_inds = np.random.choice(actions_inds, int(dact*actions_inds.shape[0]))
                actions_r.view(-1)[actions_inds] = torch.from_numpy(np.random.randint(0, actions_r.max()+1, actions_inds.shape))
            actions_r = actions_r.to(device)
            actions_c = ebatch[:,:,7:8].clone().to(device)
            labels = ebatch[:,:,7:8].clone().to(device)
            states = ebatch[:,:,8:108].clone().to(device)
            means = ebatch[:,:, 2:3].clone().to(device)
            
            return {
                'masks': masks, 
                'timesteps': timesteps, 
                'actions_h': actions_h, 
                'actions_r': actions_r, 
                'actions_c': actions_c, 
                'labels': labels, 
                'states': states,
                'means': means
            }

        batches = [get_data(batch, device, dact) for batch in batches]
        future_batches = [get_data(future_batch, device, dact, mask) for future_batch, mask in zip(future_batches, masks)]
        return batches, future_batches, thetas[0]

    def eval_prefs(model, objectives=train_objectives):
        all_accs = []
        all_accs0 = []
        all_regrets = []
        all_losses = []
        objs_repr = torch.eye(N_OBJECTS, dtype=torch.double)
        locs_repr = torch.eye(N_LOCATIONS, dtype=torch.double)

        eval_these = np.random.permutation(eval_data[:,0].unique().numpy())[:num_eval_episodes]

        with tqdm.tqdm(eval_these, desc=f'Eval Epoch', unit="trajectory") as t_eval_inds:
            for ep_ind in t_eval_inds:
                episode = eval_data[eval_data[:,0] == ep_ind].view(1,-1,d_all)
                objective_id = episode[0,0,3]
                objective = torch.tensor(objectives[objective_id])

                states = episode[:,:,8:108].clone().to(device)

                means = episode[:,:,2:3].clone().to(device)
                timesteps = episode[:,:,5:6].clone().to(device)
                actions_h = episode[:,:,6:7].clone().to(device)
                actions_r = episode[:,:,7:8].clone().to(device)
                actions_c = episode[:,:,7:8].clone().to(device)

                # state = torch.zeros((1,0,12), dtype=torch.long).to(device)
                state = torch.zeros((1,0,100), dtype=torch.long).to(device)
                prev_states = [torch.zeros((1,0,100), dtype=torch.long).to(device) for _ in range(K)]
                
                action_h = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_h = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]
                
                action_c = torch.zeros((1,0,1), dtype=torch.long).to(device)
                prev_actions_c = [torch.zeros((1,0,1), dtype=torch.long).to(device) for _ in range(K)]

                action_r = torch.zeros((1,0,1), dtype=torch.long).to(device)

                timestep = torch.zeros((1,0,1), dtype=torch.long).to(device)

                rmap = objs_repr @ objective.T @ locs_repr

                placement_map = torch.zeros_like(rmap)[0]

                min_reward = -1

                episode_regrets = []
                episode_losses = []
                for i in range(states.shape[1]-1):

                    for pid, pstate in enumerate(prev_states):
                        ind = max((i-(pid+1)), 0)
                        prev_states[pid] = torch.cat((prev_states[pid], states[0, ind].view(1,1,-1)), 1)
                    
                    for aid, pah in enumerate(prev_actions_h):
                        ind = max((i-(aid+1)), 0)
                        prev_actions_h[aid] = torch.cat((prev_actions_h[aid], actions_h[0, ind].view(1,1,-1)), 1)

                    for aid, pac in enumerate(prev_actions_c):
                        ind = max((i-(aid+1)), 0)
                        prev_actions_c[aid] = torch.cat((prev_actions_c[aid], actions_c[0, ind].view(1,1,-1)), 1)

                    state = torch.cat((state, states[0,i].view(1,1,-1)), 1)
                    action_h = torch.cat((action_h, actions_h[0,i].view(1,1,-1)), 1)
                    action_c = torch.cat((action_c, actions_c[0,i].view(1,1,-1)), 1)

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

                    if config['policy'] == 'BC':
                        pred_probs = out['action_r_preds'].reshape(-1,d_all)
                        action_r_pred = torch.argmax(pred_probs, -1)
                    
                        loss = loss_fn[0](
                            pred_probs, actions_c[:,i,:].view(-1)
                        )
                        acc = torch.sum((action_r_pred == actions_c[:,i,:].view(-1))) / actions_c[:,i,:].view(-1).shape[0]
                    else:
                        pref_pred = out['theta_preds']
                        pred_probs = (objs_repr[action_h[:,-1].cpu()-objects_start] @ pref_pred[0,-1].view(N_LOCATIONS,N_OBJECTS).T.cpu().to(torch.double)) #+ placement_map
                        action_r_pred = torch.argmax(pred_probs + placement_map).to(device)+locations_start
                        
                        loss = loss_fn[1](
                            pred_probs.view(1,N_LOCATIONS).to(device), actions_c[:,i,:].view(-1) - locations_start
                        )
                        
                        acc = torch.sum(((action_r_pred) == actions_c[:,i,:].view(-1))) / actions_c[:,i,:].view(-1).shape[0]

                    action_r = torch.cat((action_r, action_r_pred.view(1,1,-1)), 1)
                    episode_losses.append(loss.item())

                    placement_map[action_c[0,-1].item()-locations_start] = -np.inf

                    robot_reward = rmap[action_h[0,-1].item()-objects_start, action_r[0,-1].item()-locations_start].item() if action_r[0,-1].item()-locations_start < N_LOCATIONS else min_reward
                    robot_reward = min_reward if np.isinf(robot_reward) else robot_reward

                    human_reward = rmap[action_h[0,-1].item()-objects_start, action_c[0,-1].item()-locations_start].item()

                    rmap[:, action_c[0,-1].item()-locations_start] = -np.inf

                    episode_regrets.append(human_reward - robot_reward)

                acc = (torch.sum(action_r == (action_c)) / action_c.shape[1]).item()
                acc0 = (torch.sum(action_r[:,1:] == (action_c[:,:-1])) / action_c.shape[1]).item()

                all_accs.append(acc)
                all_accs0.append(acc0)
                all_regrets.append(np.mean(episode_regrets))
                all_losses.append(np.mean(episode_losses))
                t_eval_inds.set_postfix(acc=np.mean(all_accs), acc0=np.mean(all_accs0), loss=np.mean(all_losses), regret=np.mean(all_regrets))

        return {
            'accs': np.mean(all_accs),
            'regrets': np.mean(all_regrets),
            'losses': np.mean(all_losses)
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
        n_positions = config.get('n_positions'), #, 4*140), #1024),
        d_hidden = config.get('d_hidden'),#, 128),
        n_pref_layers = config.get('n_pref_layers'), #, 3),
        n_head = config.get('n_head'), #, 1),
        n_inner = None if not config.get('n_inner') else config.get('n_inner'), #, 4*128),
        activation_function = activation_function, #, 'relu'),
        resid_pdrop = config.get('resid_pdrop'), #, .1),
        embd_pdrop = config.get('embd_pdrop'), #, .1),
        attn_pdrop = config.get('attn_pdrop'), #, .1),
    ).to(device)

    print(count_parameters(model))
    if config['load_from_state']:
        params = torch.load(f'{models_path}/{config["state_name"]}.pt')
        model.load_state_dict(params)

    optimizer1 = torch.optim.AdamW(
        list(model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    optimizer2 = torch.optim.AdamW(
        list(model.pred_action.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, factor=0.5)

    early_stopping = EarlyStopping(
        patience=10, 
        verbose=True, 
        delta=1e-2, 
        path=models_path.joinpath('checkpoint.pt'), 
        trace_func=print
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=[optimizer1, optimizer2],
        batch_size=bsz,
        get_batch = get_batch,
        scheduler=scheduler,
        loss_fn=loss_fn,
        eval_fns=[eval_prefs],
        k=config['K'],
        d_all=d_all,
        device=device,
        n_objects=N_OBJECTS,
        n_locations=N_LOCATIONS,
        policy=policy
    )

    if log_to_wandb:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project=config["wandb_name"],
            # Track hyperparameters and run metadata
            config=config
        )
    for iter in range(config['max_iters']):
        outputs = trainer.train_iteration(num_steps=config['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        early_stopping(outputs['evaluation/losses'], model)
        if early_stopping.early_stop:
            break
        
        if log_to_wandb:
            wandb.log(outputs)
    
    with open(models_path.joinpath('logs.pkl'), 'wb') as f:
        pickle.dump(outputs, f)

class Trainer:
    ### grabbed from: https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/decision_transformer/training/trainer.py
    ### MIT LICENSE 
    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, early_stopping=None, k=None, d_all=None, n_objects=None, n_locations=None, policy='IRL', device='cpu'):
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

        self.start_time = time.time()
        self.k = k

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        train_accs = []
        state_accs = []
        ah_accs = []
        pstate_accs = []
        pah_accs = []
        pac_accs = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        with tqdm.tqdm(range(num_steps), desc=f'Train Epoch', unit="batch") as tepoch:
            for i in tepoch:
                train_loss, accs = self.train_step(i)

                train_losses.append(train_loss)
                train_accs.append(accs['ar'])
                if self.scheduler is not None:
                    self.scheduler.step(np.mean(train_loss))
                tepoch.set_postfix(loss=np.mean(train_losses), acc=np.mean(train_accs))
            
        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/ar_acc'] = np.mean(train_accs)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self, i):
        states, actions_h, actions_r, actions_c, timesteps, attention_mask, objs = self.get_batch(self.batch_size)
        states_target, actions_h_target, actions_r_target, actions_c_target = torch.clone(states), torch.clone(actions_h), torch.clone(actions_c), torch.clone(actions_c)

        states_preds, actions_h_preds, actions_r_preds, actions_c_preds = self.model.forward(
            states, actions_h, actions_r, actions_c, timesteps, masks=None, attention_mask=attention_mask
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            states_preds, actions_h_preds, actions_r_preds, actions_c_preds,
            states_target[:,1:], actions_h_target, actions_r_target, actions_c_target
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

class SequenceTrainer(Trainer):
    ### grabbed from: https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/decision_transformer/training/seq_trainer.py
    ### MIT LICENSE 
    def train_step(self, i):
        batches, future_batches, objectives = self.get_batch(self.batch_size, step=i)
        actions_r_target = torch.clone(batches[0]['labels'])

        prev_states = [batch['states'] for batch in batches[1:]]
        prev_actions_h = [batch['actions_h'] for batch in batches[1:]]
        prev_actions_c = [batch['actions_c'] for batch in batches[1:]]

        out = self.model.forward( # maybe change this to match better with outputs
            batches[0]['states'], 
            batches[0]['actions_h'], 
            prev_states,
            prev_actions_h,
            prev_actions_c,
            batches[0]['timesteps'], 
            attention_mask=batches[0]['masks'], 
            strategy=objectives,
            c = batches[0]['means']
        )

        if self.policy != 'IRL':
            preds = out['action_r_preds'].reshape(-1,self.d_all)

            loss = self.loss_fn[0](
                preds, actions_r_target.view(-1)
            )
            acc = torch.sum((torch.argmax(preds, -1) == actions_r_target.view(-1))) / actions_r_target.view(-1).shape[0]

        else:
            loss = torch.tensor(0).to(self.device).to(torch.float)
            for bi,batch in enumerate(future_batches):
                if (batch['masks'] == 1).sum():
                    bsz, sql = batches[0]['states'].shape[0], batches[0]['states'].shape[1]
                    label = batch['labels']
                    preds = out['theta_preds'].view(bsz,self.n_locations,self.n_objects)[torch.arange(bsz), :, (batch['actions_h']-objects_start).view(-1)].view(bsz,self.n_locations)
                    loss += self.loss_fn[1](
                            preds[(batch['masks'] != 1).view(-1),:], (label.view(-1) - locations_start)[(batch['masks'] != 1).view(-1)]
                        ) / (batch['masks'] == 1).sum()
            loss /= len(future_batches)
            acc = torch.sum((torch.argmax(preds, -1)+locations_start == actions_r_target.view(-1))) / actions_r_target.view(-1).shape[0]


        self.optimizers[0].zero_grad()
        loss.backward()
        self.optimizers[0].step()
        
        return loss.detach().cpu().item(), {
            'ar': acc.detach().cpu().item(),
            }
    
@hydra.main(version_base=None, config_path='configs', config_name='sweep-train-dt-0')
def main(config):
    dataset_pfx = 'datasets'
    dataset = 'train'
    experiment(config, dataset_pfx, dataset)

if __name__ == '__main__':
    main()