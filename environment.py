import torch
from models_rlhf import LSTMModel
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
# import torchmetrics
from reward import calculate_reward
import gym
from gym import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
# from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
# from sb3_contrib.common.wrappers import ActionMasker


def load_data(path="./data/rlhf/", dataset="causal0124"):
    #测试用的养护数据集
    """Load citation network dataset (cora only for now)"""#cora dataset: cites are edges of graphs, content is features, like id, word, class
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))#原content里的数据矩阵
    features = idx_features_labels[:, :].astype(np.float64)#[500,n_sample]
    # labels = encode_onehot(idx_features_labels[:, -1])

    # build graph|adj
    n_sample = 1918#测试样本数
    time_len =100#共100天
    nlabel = 13#特征数
    indices = [i for i in range(0,100,2)]#隔2天取一个时间步
    features = (features.transpose(1,0))[:,[range(i,nlabel*time_len,nlabel) for i in range(nlabel)]]#[n_sample,500] ->[n_sample,nlabel,time_len]
    features = features.transpose(0,2,1)[:,indices,:]

    # Flatten the time step and feature dimensions
    data_flattened = features.reshape(n_sample, -1)  # The '-1' tells numpy to calculate the size of the second dimension

    # Create a DataFrame
    df = pd.DataFrame(data_flattened)

    return df

# def load_data_id(path="./data/rlhf/", dataset="causal0124"):#修改ing
#     """Load citation network dataset (cora only for now)"""#cora dataset: cites are edges of graphs, content is features, like id, word, class
#     print('Loading {} dataset...'.format(dataset))

#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))#原content里的数据矩阵
#     features = idx_features_labels[:, :].astype(np.float64)#[500,n_sample]
#     # labels = encode_onehot(idx_features_labels[:, -1])

#     # build graph|adj
#     n_sample = 1918
#     time_len =100
#     nlabel = 13
#     indices = [i for i in range(0,100,2)]
    
#     features = (features.transpose(1,0))[:,[range(i,nlabel*time_len,nlabel) for i in range(nlabel)]]#[n_sample,500] ->[n_sample,nlabel,time_len]
#     id_list = features[:,12,-1]
#     features = features.transpose(0,2,1)[:,indices,:]

#     # Flatten the time step and feature dimensions
#     data_flattened = features.reshape(n_sample, -1)  # The '-1' tells numpy to calculate the size of the second dimension

#     # Create a DataFrame
#     df = pd.DataFrame(data_flattened)

#     return df,id_list
def predict(model,features):
    #环境 即调用lstm模型
    features = torch.FloatTensor(features)
    ob_x_len = 50
    ob_y_len = 40
    pre_x_len = 10

    ob_x = features.reshape(1,50,13)#输入全部50天*13个feature的数据

    for i in range(ob_y_len,ob_x_len):
        #遮掉需要预测的时间范围的路面性能结果，第5个特征
        ob_x[0,i,4] = -1
    output = model(ob_x)
    # print(output.shape)
    for i in range(ob_y_len,ob_x_len):
        ob_x[0,i,4] = output[0,i-ob_y_len]
    return ob_x.reshape(50,13).detach().numpy()

class PavementMaintenanceEnv(gym.Env):
    def __init__(self, dataset, model):
        super(PavementMaintenanceEnv, self).__init__()

        self.dataset = dataset
        self.current_state = self.reset()
        self.model = model
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(400)  # 7 actions * 50 time steps
        self.valid_actions = [True for _ in range(400)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50, 13), dtype=np.float32)

    def step(self,action_number):
        # Execute one time step within the environment
        # assert self.valid_actions[action_number], "Invalid action"
        state = self.current_state
        # self.set_valid_actions(state)
        dtype = state[0][5]
        size = state[0][6]
        
        origin_process_date =50#初始病害养护时间
        for i in range(50):
            if state[i][6] - 0.0 < 0.01:
                origin_process_date = i
                break

        action_value = [1.2, 1.1, 1, 1.05, 0.9, 0.85, 0.8, 0.0]
        action_time, action_type = divmod(action_number, 8)

        # print(action_type, action_time)
        # print(action_number)
        origin_method = state[-1][7].copy()
        performance_norm = 1.8873107275319592
        dtype_norm = 2.5
        state[origin_process_date:,4] += origin_method*dtype*dtype_norm/performance_norm#恢复到病害没有处理过的状态

        
        response = state[0][11] * 134#初始响应时长
        state[0][11] = (response + (origin_process_date - action_time) * 3) / 134
        method = action_value[action_type]
        if action_time == 50 or action_type == 7:
            method = 0
            state[:,7] = method
            state[:,5] = dtype
            state[:,6] = size
        else:
            method = action_value[action_type]
            state[:action_time,5] = dtype
            state[:action_time,6] = size
            state[:action_time,7] = 0
            state[action_time:,4] -= method * dtype * dtype_norm/performance_norm
            state[action_time:,5] = 0 
            state[action_time:,6] = 0
            state[action_time:,7] = method
        # ... apply the action, calculate the reward, etc. ...

        state_up = predict(self.model, state)
        reward = calculate_reward(state_up)
        done = True  # The episode ends after one step
        self.current_state = state_up
        
        return state_up, reward, done, {} 

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_state = self.dataset.sample().values.reshape(50,13)
        return self.current_state

class PavementMaintenanceEnv_id(gym.Env):
    def __init__(self, dataset, model):
        super(PavementMaintenanceEnv_id, self).__init__()

        self.dataset = dataset
        self.current_state, self.id = self.reset()
        self.model = model
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(400)  # 7 actions * 50 time steps
        self.valid_actions = [True for _ in range(400)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50, 13), dtype=np.float32)

    def step(self,action_number):
        # Execute one time step within the environment
        # assert self.valid_actions[action_number], "Invalid action"
        state = self.current_state
        # self.set_valid_actions(state)
        dtype = state[0][5]
        size = state[0][6]
        
        origin_process_date =50#初始病害养护时间
        for i in range(50):
            if state[i][6] - 0.0 < 0.01:
                origin_process_date = i
                break

        action_value = [1.2, 1.1, 1, 1.05, 0.9, 0.85, 0.8, 0.0]
        action_time, action_type = divmod(action_number, 8)

        # print(action_type, action_time)
        # print(action_number)
        origin_method = state[-1][7]
        performance_norm = 1.8978367880882472
        dtype_norm = 2.5
        state[origin_process_date:,4] += origin_method*dtype*dtype_norm/performance_norm#恢复到病害没有处理过的状态

        
        response = state[0][11] * 134#初始响应时长
        state[0][11] = (response + (origin_process_date - action_time) * 3) / 134
        method = action_value[action_type]
        if action_time == 50 or action_type == 7:
            method = 0
            state[:,7] = method
            state[:,5] = dtype
            state[:,6] = size
        else:
            method = action_value[action_type]
            state[:action_time,5] = dtype
            state[:action_time,6] = size
            state[:action_time,7] = 0
            state[action_time:,4] -= method * dtype * dtype_norm/performance_norm
            state[action_time:,5] = 0 
            state[action_time:,6] = 0
            state[action_time:,7] = method
        # ... apply the action, calculate the reward, etc. ...
        # print("method:" + str(method * dtype * dtype_norm/performance_norm))
        # print("actual date:" + str(origin_process_date))
        # print("action date:" + str(action_time))
        state_up = predict(self.model, state)
        reward = calculate_reward(state_up)
        done = True  # The episode ends after one step
        self.current_state = state_up
        
        return state_up, reward, done, {} 

    def reset(self):
        # Reset the state of the environment to an initial state
        sample_state = self.dataset.sample()

        self.current_state = sample_state.values.reshape(50,13)
        self.id = sample_state.index.item()
        return self.current_state, self.id

class MaskedPPOPolicy(ActorCriticPolicy):
    def _make_action(self, obs):
        # Start by calling the original _make_action method
        action, states, values = super()._make_action(obs)

        # Apply the mask to the action
        action_mask = self.env.valid_actions
        action = [a for a, valid in zip(action, action_mask) if valid]

        return action, states, values

    # dataset = dataset.cuda()



'''def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


env = ...  # Initialize env
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
# with ActionMasker. If the wrapper is detected, the masks are automatically
# retrieved and used when learning. Note that MaskablePPO does not accept
# a new action_mask_fn kwarg, as it did in an earlier draft.
model = MaskedPPOPolicy(MaskableActorCriticPolicy, env, verbose=1)
model.learn()

# Note that use of masks is manual and optional outside of learning,
# so masking can be "removed" at testing time
model.predict(observation, action_masks=valid_action_array)'''



