# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import transformers

from trajectory_gpt import GPT2Model
from layers import OneHotEmbedding

def construct_layers(d_layers, relu=True, pdrop=0.1):
    if len(d_layers) < 3:
        return nn.Sequential(nn.Linear(d_layers[0], d_layers[1]))
    first = nn.Sequential(nn.Linear(d_layers[0], d_layers[1][0]))
    hidden = nn.Sequential(
        *[nn.Sequential(
            *[nn.Linear(sz[0], sz[1])] + 
            ([nn.ReLU()] if relu is not None else []) + 
            ([nn.Dropout(pdrop)] if pdrop else [])
        ) for sz in d_layers[1:-1]
    ])
    last = nn.Sequential(
        nn.Linear(d_layers[-2][1], d_layers[-1])
    )
    return first + hidden + last

class PreferenceMLP(nn.Module):    
    def __init__(self, **kwargs):
        super().__init__()

        self.sz_vocab = kwargs.pop('sz_vocab') #, 208)
        self.d_embed = kwargs.pop('d_embed')
        self.K = kwargs.pop('K') # 0)
        n_pref_layers = kwargs.pop('n_pref_layers')#, 5)
        n_act_layers = kwargs.pop('n_act_layers') #, 5)
        activation_function = kwargs.pop('activation_function')
        self.pdrop = kwargs.pop('pdrop') #, 0)
        self.d_hidden = kwargs.pop('d_hidden') #, 16)
        self.n_objects = kwargs.pop('n_objects')
        self.n_locations = kwargs.pop('n_locations')
        sz_state = kwargs.pop('sz_state')

        n_states = sz_state*(self.K+1)
        n_ah_vocabs = (self.K+1)
        n_ar_vocabs = self.K

        sz_z = int((self.n_objects*self.n_locations) // 1)

        if not self.d_embed:
            self.vocab = OneHotEmbedding(self.sz_vocab)
            self.d_embed = self.sz_vocab
        else:
            self.vocab = nn.Embedding(self.sz_vocab, self.d_embed)

        sz_pref_in = self.d_embed*n_states + (n_ah_vocabs+n_ar_vocabs)*self.d_embed
        assert(sz_pref_in // (2**(n_pref_layers+1)))
        
        sz_pref_layers = [sz_pref_in] + [(self.d_hidden, self.d_hidden) for i in range(n_pref_layers)] + [sz_z]
        self.pred_pref = construct_layers(sz_pref_layers, activation_function, self.pdrop)

        state_in = sz_state*1
        action_in = 1
        sz_act_in = sz_z + self.d_embed*state_in + self.d_embed*action_in 
        assert(sz_act_in // (2**(n_act_layers+1)))
        sz_act_layers = [sz_act_in] + [(self.d_hidden, self.d_hidden) for i in range(n_act_layers)] + [self.sz_vocab]
        self.pred_action = construct_layers(sz_act_layers, activation_function, self.pdrop)

        self.pred_state = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.pdrop),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(self.pdrop),
            nn.Linear(self.d_hidden, 208 * 100)
        )

        self.pred_ah = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.pdrop),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(self.pdrop),
            nn.Linear(self.d_hidden, 208)
        )

        self.pred_ac = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.pdrop),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(self.pdrop),
            nn.Linear(self.d_hidden, 208)
        )

    def get_features(self, x, sz_batch, sz_seq):
        features = self.vocab(x).view(sz_batch,sz_seq, -1) if x is not None else x
        return features

    def get_pref(self, features):# state_features, ah_features, prev_state_features, prev_ah_features, prev_ac_features, prev_prev_state_features, prev_prev_ah_features, prev_prev_ac_features):
        return self.pred_pref(features)

    def get_action(self, *args): #pref, state_features, ah_features):
        features = torch.cat([arg for arg in args if arg is not None], -1)
        return self.pred_action(features)

    def forward(self, state, action_h, prev_states, prev_actions_h, prev_actions_c, timesteps, c=None, attention_mask=None, strategy=None, irl=False):
        sz_batch, sz_seq = state.shape[0], state.shape[1]

        state_features = [self.get_features(state, sz_batch, sz_seq)]
        action_h_features = [self.get_features(action_h, sz_batch, sz_seq)]
        prev_states_features = [self.get_features(prev_state, sz_batch, sz_seq) for prev_state in prev_states]
        prev_actions_h_features = [self.get_features(prev_action_h, sz_batch, sz_seq) for prev_action_h in prev_actions_h]
        prev_actions_c_features = [self.get_features(prev_action_c, sz_batch, sz_seq) for prev_action_c in prev_actions_c]
        
        feat_ls = state_features + action_h_features + prev_states_features + prev_actions_h_features + prev_actions_c_features
        feat_ls = [feat for feat in feat_ls if feat is not None]
        features = torch.cat(feat_ls, -1)
        
        prefs = self.get_pref(features)
        acts = self.get_action(prefs, *state_features, *action_h_features)

        return {
            'action_r_preds': acts, 
            'theta_preds': prefs
        }

class PreferenceTransformer(PreferenceMLP):
    def __init__(
            self, 
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.time_embed = nn.Embedding(1000, self.d_hidden)

        n_positions = kwargs.pop('n_positions')
        config = transformers.GPT2Config(
            vocab_size = 1, # doesn't matter -- we don't use the vocab
            n_positions = n_positions,
            n_ctx = n_positions,
            n_embd = kwargs.pop('d_hidden'),
            n_layer = kwargs.pop('n_pref_layers'),
            n_head = kwargs.pop('n_head'),
            n_inner = kwargs.pop('n_inner'),
            activation_function = kwargs.pop('activation_function'),
            resid_pdrop = kwargs.pop('resid_pdrop'),
            embd_pdrop = kwargs.pop('embd_pdrop'),
            attn_pdrop = kwargs.pop('attn_pdrop'),
        )
        
        self.embed_ln = nn.LayerNorm(self.d_hidden)
        self.dropout = nn.Dropout(self.pdrop)

        self.pred_pref = GPT2Model(config)
        self.resize_pref = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.pdrop),
            nn.Linear(self.d_hidden, self.n_locations*self.n_objects),
        )

        self.resize_state = nn.Sequential(
            nn.Linear(self.sz_vocab * 10 * 10, self.d_hidden)
        )

        self.resize_ah = nn.Sequential(
            nn.Linear(self.sz_vocab, self.d_hidden)
        )

        self.resize_ac = nn.Sequential(
            nn.Linear(self.sz_vocab, self.d_hidden)
        )

    def forward(self, state, action_h, prev_states, prev_actions_h, prev_actions_c, timesteps, c=None, attention_mask=None, strategy=None, irl=False):
        sz_batch, sz_seq = state.shape[0], state.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((sz_batch, sz_seq), dtype=torch.long).to(state.device)

        all_timesteps = [timesteps-i for i in range(len(prev_states)+1)]
        for ti,t in enumerate(all_timesteps):
            t[t < 0] = 0
            all_timesteps[ti] = t

        time_embeddings = [self.time_embed(t.to(torch.int)).view(sz_batch, sz_seq, -1) for t in all_timesteps]

        state_features = [self.resize_state(self.get_features(state, sz_batch, sz_seq)) + time_embeddings[0]]
        action_h_features = [self.resize_ah(self.get_features(action_h, sz_batch, sz_seq)) + time_embeddings[0]]
        prev_states_features = [self.resize_state(self.get_features(prev_state, sz_batch, sz_seq)) + time_embeddings[psi+1] for psi, prev_state in enumerate(prev_states)]
        prev_actions_h_features = [self.resize_ah(self.get_features(prev_action_h, sz_batch, sz_seq)) + time_embeddings[pai+1] for pai, prev_action_h in enumerate(prev_actions_h)]
        prev_actions_c_features = [self.resize_ac(self.get_features(prev_action_c, sz_batch, sz_seq)) + time_embeddings[pai+1] for pai, prev_action_c in enumerate(prev_actions_c)]

        stacked_inputs = torch.stack(
            (*prev_states_features, *prev_actions_h_features, *prev_actions_c_features, *state_features, *action_h_features), dim=1
        ).permute(0, 2, 1, 3).reshape(sz_batch, (2+(3*self.K))*sz_seq, self.d_hidden)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            [attention_mask]*(stacked_inputs.shape[1]//sz_seq), dim=1
        ).permute(0, 2, 1).reshape(sz_batch, (2+(3*self.K))*sz_seq)

        transformer_outputs = self.pred_pref(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        
        x = x.reshape(sz_batch, sz_seq, stacked_inputs.shape[1] // sz_seq, self.d_hidden).permute(0, 2, 1, 3)
       
        x = x[:,-1] # p(a_r | a_h, s, k)
        prefs = self.resize_pref(x)

        acts = self.get_action(prefs, self.get_features(state, sz_batch, sz_seq), self.get_features(action_h, sz_batch, sz_seq))
        
        state_pred = self.pred_state(x)
        ah_pred = self.pred_ah(x)
        ac_pred = self.pred_ac(x)

        return {
            'action_r_preds': acts, 
            'theta_preds': prefs,
            'state_preds': state_pred,
            'action_h_preds': ah_pred,
            'action_c_preds': ac_pred,
        }