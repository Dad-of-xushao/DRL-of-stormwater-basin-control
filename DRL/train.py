import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
from MySwmm import swmmEnv
import collections
import torch
import torch.nn.functional as F
import os
from DQN import ReplayBuffer,DQN

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

if __name__=='__main__':
    lr = 1e-5
    num_episodes = 150
    state_dim = 6
    hidden_dim = 20
    action_dim = 11
    gamma = 0.95 
    epsilon = 0.1
    target_update = 5
    buffer_size = 1000
    minimal_size = 200
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    print('using:',device)
    inputfile=["runfile25.inp","runfile50.inp"]
    for i in range(len(inputfile)):
        inputfile[i]="model/"+inputfile[i]
    env=swmmEnv(inputfile)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                state=state[:-1]
                done = False
                while not done:
                    # depth,inflow,outflow,rainfall,tss,_,_=state
                    action = agent.take_action(state)
                    next_state, reward, done= env.step(action)
                    next_state=next_state[:-1]
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    torch.save(agent.q_net.state_dict(), 'model.pth')
    env.close()
    episodes_list = list(range(len(return_list)))
    episodes_list=np.array(episodes_list)
    episodes_list=episodes_list
    return_list=moving_average(return_list,19)
    plt.plot(episodes_list, return_list)
    plt.xlabel('time')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('Swmm'))
    plt.savefig("returns")
    plt.show()
