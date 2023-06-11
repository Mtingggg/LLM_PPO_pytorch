# LLM_PPO_pytorch
Implement PPO in LLM with pytorch from scratch

## Introduction
This is a simple implementation of RLHF (Reinforcement Learning with Human Feedback) with pytorch.

Example use [huggingtweets/elonmusk](https://huggingface.co/huggingtweets/elonmusk) as base model and [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) to simulate Human Feedback.

After training on several steps, we make the base model tend to generates negative sentences.

## Example
Checkout the simple [example](https://github.com/Mtingggg/LLM_PPO_pytorch/blob/main/example.ipynb) of this repo

## Train
```py
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead

from environment import Chat
from actor_critic_net import ActorCritic
from agent import PPO
from utils import trainer

tokenizer = AutoTokenizer.from_pretrained('huggingtweets/elonmusk')
model = AutoModelWithLMHead.from_pretrained('huggingtweets/elonmusk').to(device)
feedback_pipe = pipeline('sentiment-analysis',
                        model="cardiffnlp/twitter-roberta-base-sentiment",
                        tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
                        return_all_scores=True, 
                        device=device)

####### initialize environment hyperparameters ######

max_ep_len = 20    # max timesteps in one episode
obs_base = ['I think Tesla is', 
            'I think dogecoin is', 
            'I think BTC is', 
            'I think Twitter is', 
            'I think ElonMusk is']    # init observations that will use to run the episodes

################ PPO hyperparameters ################
epochs = 50    # update policy for K epochs
eps_clip = 0.2    # clip parameter for PPO
gamma = 1    # discount factor

lr_actor = 5e-6    # learning rate for actor network
lr_critic = 5e-6    # learning rate for critic network

################## Trainer Setting ##################
max_training_timesteps = 30000
print_freq = 100
update_timestep = max_ep_len * 100
save_model_freq = max_ep_len * 100
checkpoint_path = "PPO_model.pth"
eval_obs = 'I think Tesla is'    # observation use to check performance after model update

env = Chat(llm=model, tokenizer=tokenizer, reward_pipe=feedback_pipe, obs_base=obs_base, max_gen_len=max_ep_len)
policy = ActorCritic(model).to(device)
ppo_agent = PPO(policy=policy, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=epochs, eps_clip=eps_clip)

trainer(
    env,
    ppo_agent,
    max_training_timesteps=max_training_timesteps,
    max_ep_len=max_ep_len,
    update_timestep=update_timestep,
    save_model_timestep=save_model_freq,
    print_freq=print_freq,
    checkpoint_path=checkpoint_path,
    eval_obs=eval_obs
)
```

## Prediction
```py
from utils import Predictor

predictor = Predictor(env, policy)
predictor.load_model('your/best/model')

predictor.predict('input text')
```
