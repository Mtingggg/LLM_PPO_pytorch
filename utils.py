import torch

def trainer(
    env, 
    ppo_agent, 
    max_training_timesteps,
    max_ep_len,
    update_timestep,
    save_model_timestep,
    print_freq,
    checkpoint_path, 
    eval_obs):

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_done.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
                
                # evaluation
                with torch.no_grad():
                    state = env.reset(cur_obs_base=eval_obs)
                    for t in range(max_ep_len):
                        action = ppo_agent.select_action(state, train=False, sample=True)
                        state, reward, done, info = env.step(action)    
                        if done:
                            break
                    print('generated sentence:', env.cur_obs_base+''.join(env.predicted),'\n','score:', reward)

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_timestep == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1
        
class Predictor:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy.eval()
    
    @torch.inference_mode()
    def predict(self, inputs, max_len=-1):
        done=False
        cur_len=0
        state = self.env.reset(cur_obs_base=inputs)
        while not done:
            action, _, _ = self.policy.act(state, sample=True)
            state, reward, done, info = self.env.step(action)
            cur_len+=1
            if max_len!=-1 and cur_len>=max_len:
                break
        return self.env.cur_obs_base+''.join(self.env.predicted)
    
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()
