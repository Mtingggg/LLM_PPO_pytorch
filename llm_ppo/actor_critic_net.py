import torch

class ActorCritic(torch.nn.Module):
    def __init__(self, LLM, temperature=1, top_p=1, top_k=0):
        super(ActorCritic, self).__init__()
        # actor
        self.actor = torch.nn.Sequential(
            LLM.lm_head,
            CategoricalNet(temperature=temperature, top_p=top_p, top_k=top_k)
            )

        # critic
        self.critic = torch.nn.Sequential(
                torch.nn.Linear(LLM.config.hidden_size, 1)
#                 torch.nn.Linear(LM_model.config.hidden_size // 2, LM_model.config.hidden_size // 4),
#                 torch.nn.Linear(LM_model.config.hidden_size // 4, 1)
            )
        self.device = LLM.device

    def act(self, state, sample):
        device = state.device.type
        action_probs = self.actor(state)
        action_probs = action_probs.detach().to('cpu') # there's a bug when doing argmax on mps
        dist = torch.distributions.Categorical(action_probs)
        if sample:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs, dim=-1)
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach().to(device), action_logprob.detach().to(device), state_val.detach()
    

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    
class CategoricalNet(torch.nn.Module):
    def __init__(self, temperature=1, top_p=1, top_k=0):
        super(CategoricalNet, self).__init__()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
    def forward(self, logits):
        logits = logits/self.temperature
        
        if self.top_k>0:
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][-1]
            indices_to_remove = indices_to_remove
            logits[indices_to_remove] = -float("Inf")
            
        if 0<self.top_p<=1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            # only keep indices cumulative_probs is lower than p
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # always keep as least one token, so we set index 0 value to False
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("Inf")
            
        return torch.distributions.Categorical(logits=logits).probs
