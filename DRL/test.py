import matplotlib.pyplot as plt
import numpy as np
from MySwmm import swmmEnv
import torch
import torch.nn.functional as F
import os 
from DQN import DQN

if __name__=='__main__':
    env=swmmEnv(["model/runfile.inp"])
    lr = 1e-4
    num_episodes = 100
    state_dim = 6
    hidden_dim = 20
    action_dim = 11
    gamma = 0.9 
    epsilon = 0.1
    target_update = 5
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                    target_update, device)
    agent.q_net.load_state_dict(torch.load('model.pth'))
    state = env.reset()

    done = False
    action_list=[]
    state_list=[]
    episode_return=0
    while not done:
        action = agent.best_action(state[:-1])
        next_state, reward, done = env.step(action)
        state_list.append(state)
        action_list.append(action)
        state = next_state
        episode_return += reward  
    time_list = list(range(len(state_list)))
    time_list=np.array(time_list)/12.0
    state_list=np.array(state_list)
    action_list=np.array(action_list)
    print(episode_return)
    filename="dqn/"
    homedir = os.getcwd()#获取项目当前路径
    if not os.path.exists(homedir+'/'+filename):
        os.mkdir(homedir+'/'+filename)#创建pic文件夹，用于保存图片 
    plt.plot(time_list,action_list)
    plt.xlabel("Time(h)")
    # plt.ylabel("depth(m)")
    plt.title("action")
    plt.savefig(filename+"action.png")
    #plt.cla()
    plt.show()

    plt.plot(time_list,state_list[:,0])
    plt.xlabel("Time(h)")
    plt.ylabel("depth(m)")
    plt.title("depth")
    plt.savefig(filename+"depth.png")
    #plt.cla()
    plt.show()
    print(max(state_list[:,0]))

    plt.plot(time_list,state_list[:,1],label="inflow")
    plt.title("inflow")
    plt.xlabel("Time(h)")
    plt.ylabel("inflow(m3/s)")
    plt.savefig(filename+"inflow.png")
    #plt.cla()
    plt.show()

    plt.plot(time_list,state_list[:,2],label="outflow")
    plt.title("outflow")
    plt.xlabel("Time(h)")
    plt.ylabel("outflow(m3/s)")
    plt.savefig(filename+"outflow.png")
    #plt.cla()
    plt.show()

    plt.plot(time_list,state_list[:,3])
    plt.title("rainfall")
    plt.xlabel("Time(h)")
    plt.ylabel("rainfall(mm/hr)")
    plt.savefig(filename+"rainfall.png")
    #plt.cla()
    plt.show()

    plt.plot(time_list,state_list[:,4])
    plt.title("tss")
    plt.xlabel("Time(h)")
    plt.ylabel("tss(mg/L)")
    plt.savefig(filename+"tss.png")
    #plt.cla()
    plt.show()

    pollution=state_list[:,2]*state_list[:,4]
    len1=len(pollution)
    for i in range(1,len1):
        pollution[i]+=pollution[i-1]
    plt.plot(time_list,pollution)
    plt.title("pollution")
    plt.xlabel("Time(h)")
    plt.ylabel("pollution(g)")
    plt.savefig(filename+"pollution.png")
    #plt.cla()
    plt.show()
