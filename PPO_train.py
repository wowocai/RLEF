from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO
import torch
from models_rlhf import LSTMModel
import argparse

from environment import PavementMaintenanceEnv,MaskedPPOPolicy,load_data

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
lstm_model = LSTMModel( input_size = 13, 
                        hidden_size = 64,
                        num_layers = 3, 
                        output_size =10)
lstm_model.load_state_dict(torch.load("./372.pkl"))#environment, LSTM model for pavement performance prediction


if args.cuda:
    lstm_model.cuda()
    # dataset = dataset.cuda()

dataset = load_data()    


env = PavementMaintenanceEnv(dataset=dataset,model=lstm_model)
model = PPO(MaskedPPOPolicy, env, verbose=1,tensorboard_log="./ppo_tensorboard/",  
            n_steps=2048, 
            gamma=0.99, 
            learning_rate=0.0001, 
            ent_coef=0.05, 
            clip_range=0.2, 
            n_epochs=10, 
            batch_size=32)#MaskedPPOPolicy(MaskableActorCriticPolicy,
model.learn(total_timesteps=500000)
model.save("ppo_model_0322v3")
# obs = env.reset()
# for i in range(1000
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#       obs = env.reset()


