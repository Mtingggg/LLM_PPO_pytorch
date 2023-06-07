import gym
import random

class Chat(gym.Env):
    def __init__(self, llm, tokenizer, reward_pipe, obs_base, max_gen_len):
        self.size = max_gen_len  # The length of the sentence
        vocabs = list(dict(sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])).keys())
        self.actions = vocabs
        self.action_space = gym.spaces.Discrete(len(vocabs))   
        self.predicted = []
        self.obs_base = obs_base
        
        self.cur_obs_base = None
        self.tokenizer = tokenizer
        self.model = llm
        self.reward_pipe = reward_pipe
        self.device = llm.device
        
    def reset(self, cur_obs_base=None):
        self.predicted = []
        if cur_obs_base==None:
            self.cur_obs_base = random.sample(self.obs_base, 1)[0]
        else:
            self.cur_obs_base = cur_obs_base
        feature_dict = self.tokenizer(self.cur_obs_base,
                              return_tensors='pt',
                              add_special_tokens=False).to(self.device)
        prediction = self.model(**feature_dict, output_hidden_states=True)
        state = prediction.hidden_states[-1].squeeze(0)
        
        return state.data[-1]
    
    def step(self, action):
        pred_token = self.actions[action]
        predicted_string = self.tokenizer.convert_tokens_to_string([pred_token])
        self.predicted.append(predicted_string)
        predicted_sentence = ''.join(self.predicted)
        feature_dict = self.tokenizer(self.cur_obs_base+predicted_sentence,
                              return_tensors='pt',
                              add_special_tokens=False).to(self.model.device)
        prediction = self.model(**feature_dict, output_hidden_states=True)
        state = prediction.hidden_states[-1].squeeze(0).data[-1]
        
        done = False
        if pred_token==self.actions[-1] or len(self.predicted)>=self.size:
            done = True
        
        reward = 0
        if done:
            reward = self.get_reward(self.cur_obs_base+predicted_sentence)
        
        return state, reward, done, {'predicted_string':predicted_sentence}
    
    def get_reward(self, sentence):
        return self.reward_pipe(sentence)[0][0]['score']*10
