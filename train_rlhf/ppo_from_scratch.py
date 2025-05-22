from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import wandb

prompt = '''
Below is an instruction that describes a task. Write a response that appropriately completes the request. 
### Instruction: 
{instruction}
# ###
'''

from tqdm import tqdm
from random import randrange
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from collections import deque, namedtuple
from functools import partial

from torch.utils.data import Dataset, DataLoader
from beartype import beartype

from transformers import AutoTokenizer
from reward_model import RewardModel
from palm import PaLM

class ExperienceDataset(Dataset):
    @beartype
    def __init__(self, data, device=None):
        super().__init__()
        self.data = data
        self.device = device
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return tuple(map(lambda t: t[index].to(self.device), self.data))

def shift(t, value = 0, shift = 1):
    zeros = ()
    return F.pad(t, (*zeros, shift, -shift), value = value)
def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def pad_sequence_fixed(sequences, *args, **kwargs):
    '''
    Pads a list of variable length sequences with padding values to make them of equal length.
    '''
    first_el = sequences[0]
    has_no_dimension = first_el.ndim == 0

    # if no dimensions, add a single dimension
    if has_no_dimension:
        sequences = tuple(map(lambda t: t[None], sequences))
    out = torch.nn.utils.rnn.pad_sequence(sequences, *args, **kwargs) # from torch.rnn
    if has_no_dimension:
        out = rearrange(out, '... 1 -> ...')
    return out
def create_dataloader(data, batch_size, shuffle=True, device=None,**kwargs):
    dataset = ExperienceDataset(data, device=device)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if not exists(mask):
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))
def log_prob(prob, index):
    assert prob.shape[:2] == index.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, index[..., None])).squeeze(-1)
def masked_kl_div(prob1, prob2, mask):
    kl_div = (prob1 * (log(prob1) - log(prob2))).sum(dim=-1)
    loss = masked_mean(kl_div, mask=mask)
    return loss
def prompting_the_instruction(instructions):
    return [prompt.format(instruction=inst) for inst in instructions]
def tokenizer_customized(prompts_sequence, tokenizer):
    output_from_tokenizer = list(map(partial(tokenizer, return_tensors='pt'), prompts_sequence))
    return [sample['input_ids'].squeeze(0) for sample in output_from_tokenizer]

Memory = namedtuple('Memory', [
    'sequence',
    'prompt_mask', 
    'mask',
    'action_prob',
    'action_log_prob',
    'reward',
    'value'
])

PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', [
    'actions',
    'sequence',
    'mask',
    'prompt_mask',
    'action_logits',
    'values'
])

class ActorCritic(nn.Module):
    # define actor and critic, then write generate and forward function
    def __init__(
        self,
        palm, 
        actor_lora = True,
        critic_lora = True,
        actor_lora_r = 8,
        critic_lora_r = 8,
        actor_lora_scope = 'actor',
        critic_lora_scope = 'critic',
    ):
        super().__init__()
        self.actor_palm = palm
        self.critic = copy.deepcopy(palm)
        self.actor_lora = actor_lora
        self.critic_lora = critic_lora
        self.actor_lora_scope = actor_lora_scope if actor_lora else None
        self.critic_lora_scope = critic_lora_scope if critic_lora else None
        if self.actor_lora:
            self.actor_palm.add_finetune_params(actor_lora_scope, lora_r = actor_lora_r)
        if self.critic_lora:
            self.critic.add_finetune_params(critic_lora_scope, lora_r = critic_lora_r)
        self.value_head = nn.Sequential(
            nn.Linear(palm.dim, 1),
        )
    def actor_parameters(self):
        if not self.actor_lora:
            return self.actor_palm.parameters()

        return [
            *self.actor_palm.finetune_parameters(self.actor_lora_scope)
        ]

    def generate(
        self,
        state,
        max_seq_len = 1024,
        eos_token = 0,
        return_values = False,
        **kwargs
    ):
        actions = self.actor_palm.generate(
            max_seq_len,
            prompt = state,       
            eos_token = eos_token,     
            finetune_scope = self.actor_lora_scope,
            use_tqdm = True,
            **kwargs
        )
        sequence = torch.cat([state, actions], dim=-1)

        prompt_mask = torch.arange(sequence.size(-1), device=state.device) < state.size(-1)
        prompt_mask = repeat(prompt_mask, 'n -> b n', b=sequence.shape[0])
        action_mask = ~prompt_mask

        mask = None 
        eos_token = 0
        mask = ((sequence == eos_token).cumsum(dim=-1) == 0) # find the last place of eos_token
        action_mask = action_mask & mask

        action_logits, value = self(
            sequence,
            mask,
            return_values = return_values
        )

        return PPOActionCriticReturn(
            actions,
            sequence,
            mask,
            prompt_mask,
            action_logits,
            value
        )
        
    def forward(self, x, mask=None, return_values = True):
        action_logits = self.actor_palm(
            x,
            finetune_scope = self.actor_lora_scope
        )
        if not return_values:
            return action_logits, None
        critic_embeds = self.critic(
            x, 
            return_only_embedding = True,
            finetune_scope = self.critic_lora_scope
        )
        critic_embeds = shift(critic_embeds, shift=1)
        # calculate the values only on action tokens
        critic_embeds = masked_mean(critic_embeds, mask=mask, dim=1)
        values = self.value_head(critic_embeds)
        values = values.squeeze(dim=-1)
        return action_logits, values
        

class PPOTrainer(nn.Module):
    def __init__(
        self,
        prompt_token_ids,
        actor_critic,
        reward_model,
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased"),
        batch_size = 16,
        epochs = 1,
        kl_div_loss_weight = 0.1,
        eps_clip = 0.2,
        actor_lr = 1e-4,
        critic_lr = 1e-4,
    ): 
        super().__init__()
        self.prompt_token_ids = prompt_token_ids
        self.actor_critic = actor_critic
        self.reward_model = reward_model
        self.tokenizer =tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.kl_div_loss_weight = kl_div_loss_weight
        self.eps_clip = eps_clip
        self.actor_optim = torch.optim.Adam(actor_critic.actor_palm.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(actor_critic.critic.parameters(), lr=critic_lr)
        self.device = 'cuda'
        self.num_prompts = len(prompt_token_ids)
        self.pad_value = 0
    def calculate_generalized_advantage_estimator(self, ):
        pass
    def ppo_update(self, memories):
        all_memories_stacked_and_padded = list(map(partial(pad_sequence_fixed, batch_first = True), zip(*memories)))
        dataloader = create_dataloader(all_memories_stacked_and_padded, batch_size=self.batch_size, device=self.device)
        self.actor_critic.train()
        for _ in range(self.epochs):
            for (
                    sequences,
                    prompt_masks,
                    masks,
                    old_action_prob,
                    old_action_log_probs,
                    reward,
                    old_values
                ) in dataloader:
                # in this for loop:
                # step 0: get new action and value from actor_critic model by pass the sequence
                # step 1: get old_action_prob, action_prob for kl divergence.
                # step 2: get reward = reward - kl
                # step 3: calculate advantage
                # step 4: get old_action_log_probs and action_log_prob for ratio
                # step 5: actor loss: ratio, advantage; critic loss: old_value, value

                # get action_prob, values, action_log_prob
                action_masks = ~prompt_masks & masks
                action_logits, values = self.actor_critic(sequences, mask=action_masks)
                action_logits = shift(action_logits, shift = 1)
                action_prob = action_logits.softmax(dim=-1)

                action_len = old_action_log_probs.shape[-1]
                action_log_probs = log_prob(action_prob, index=sequences)
                action_log_probs = action_log_probs[:, -action_len:]

                # kl-based reward
                kl_penalty = .0
                if self.kl_div_loss_weight > 0:
                    kl_penalty = masked_kl_div(prob1 = action_prob, prob2 = old_action_prob, mask = action_masks) * self.kl_div_loss_weight

                reward = reward - kl_penalty

                # calculate ratio and advantage
                ratio = (action_log_probs - old_action_log_probs).exp()
                assert reward.shape == old_values.shape
                advantages = reward - old_values
                advantages = advantages.unsqueeze(dim=-1)

                # calculate actor loss: min(ratio*A, clip(ratio, -eps, eps)*A)
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss =  - torch.min(surr1, surr2).mean()
                actor_loss.backward()
                self.actor_optim.step()
                self.actor_optim.zero_grad()

                # calculate critic loss
                critic_loss = torch.mean((values - old_values) ** 2)
                critic_loss.backward()
                self.critic_optim.step()
                self.critic_optim.zero_grad()

                # log loss to wandb
                wandb.log({
                    'actor_loss': actor_loss.item(),
                    'critic_loss': critic_loss.item()
                })

    def train(
            self,
            num_episodes=5000,
            max_timesteps = 500,
            update_timesteps = 50,
            max_seq_len=2048,
            temperature=1
        ):
        device = self.device
        eos_token = self.pad_value

        time = 0
        memories = deque([])
        for eps in tqdm(range(num_episodes), desc = 'episodes'):
            for timestep in (range(max_timesteps)):
                time += 1
                rand_prompt_index = randrange(0, self.num_prompts)
                state = self.prompt_token_ids[rand_prompt_index] # randomly get a prompt
                state = state.to(self.device)
                # remove padding from state
                state_mask = state != self.pad_value
                state = state[state_mask]
                # Check if model is wrapped
                if hasattr(self.actor_critic, "module"):
                    generate = self.actor_critic.module.generate
                else:
                    generate = self.actor_critic.generate
                (
                    actions,
                    sequence,
                    mask,
                    prompt_mask,
                    action_logits,
                    value
                ) = generate(
                    rearrange(state, 'n -> 1 n'),
                    max_seq_len = max_seq_len,
                    eos_token = eos_token,
                    temperature = temperature,
                    return_values = True
                )
                action_logits = shift(action_logits, shift=1)
                action_prob = action_logits.softmax(dim=-1)
                action_len = actions.shape[-1]
                action_log_prob = log_prob(action_prob, index = sequence)
                action_log_prob = action_log_prob[:,-action_len:]

                actions = rearrange(actions, '1 ... -> ...')
                # get mask and prompt mask
                prompt_length = len(state)
                prompt_mask = torch.arange(sequence.shape[-1], device = self.device) < prompt_length
                # sequence = rearrange(sequence, 'n -> 1 n')
                prompt_mask = rearrange(prompt_mask, 'n -> 1 n')
                mask = default(mask, lambda: torch.ones(sequence.shape, dtype = torch.bool, device = self.device))

                reward = self.reward_model(
                    sequence,
                    prompt_mask = prompt_mask,
                    mask = mask,
                    sample = True
                )
                detach_to_cpu_ = lambda t: rearrange(t.detach().cpu(), '1 ... -> ...')
                memories.append(Memory(*map(detach_to_cpu_, (
                    sequence,
                    prompt_mask,
                    mask,
                    action_prob,
                    action_log_prob,
                    reward,
                    value
                ))))
                # update the weights, and the memory is cleaned up.
                if time % update_timesteps == 0:
                    self.ppo_update(memories)
                    memories.clear()
                    torch.save(self.actor_critic.state_dict(), 'ppo_model_weights.pt')
   


if __name__ == '__main__':     
    wandb.init(project="ppo_training_2", config={})  
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = load_dataset("vicgalle/alpaca-gpt4")
    prompts = dataset['train']['instruction']
    prompts_token_ids = tokenizer_customized(prompts, tokenizer)

    palm = PaLM(
        num_tokens = 32768,
        dim = 512,
        depth = 12,
        flash_attn = True # https://arxiv.org/abs/2205.14135
    )

    ac_model = ActorCritic(palm=palm).cuda()
    reward_model = RewardModel(
        palm,
        num_binned_output = 5 # say rating from 1 to 5
    ).cuda()
    
    ppo_trainer = PPOTrainer(prompt_token_ids=prompts_token_ids, actor_critic=ac_model, reward_model=reward_model, tokenizer=tokenizer)

    ppo_trainer.train()






