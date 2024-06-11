from MySwmm import swmmEnv
import torch
import torch.nn.functional as F
from DQN import DQN
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from multiprocessing import Pool

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


def rbc_easy(state):
    return 10

def rbc_tss(state):
    depth,inflow,outflow,rainfall,tss,action,_=state
    if tss>15 and depth<2.5:
        OR=0
    else:
        OR=10
    return OR

def rbc_depth(state):
    depth,inflow,outflow,rainfall,tss,action,_=state
    if depth<2:
        OR=0
    if depth>=2:
        OR=10
    if 1<depth<2 and inflow<0.01:
        OR=5
    if depth<1 and inflow<0.01:
        OR=10
    return OR

def dqn(state):
    action = agent.best_action(state[:-1])
    return action

def get_result(fun):
    env=swmmEnv(["model/runfile.inp"])
    state = env.reset()
    done = False
    action_list=[]
    state_list=[]
    episode_return=0
    while not done:
        action =fun(state)
        next_state, reward, done = env.step(action)
        state_list.append(state)
        action_list.append(action)
        state = next_state
        episode_return += reward  

    state_list=np.array(state_list)
    action_list=np.array(action_list)
    time_list = list(range(len(state_list)))
    time_list =np.array(time_list)/12.0


    hmax=2.5
    depth=state_list[:,0]
    depthOverFlow=np.maximum(0,depth-hmax)
    overflow=np.sum(depthOverFlow)
    # print("overflow:",overflow)

    outflow=state_list[:,2]
    peakOutflow=np.max(outflow)
    # print("peakOutflow:",peakOutflow)
    
    pollution=state_list[:,2]*state_list[:,4]
    len1=len(pollution)
    for i in range(1,len1):
        pollution[i]+=pollution[i-1]
    TSSload=pollution[-1]
    # print("TSSload:",TSSload)

    controlEffort=0
    len1=len(action_list)
    for i in range(1,len1):
        delta=action_list[i]-action_list[i-1]
        delta=delta*delta
        controlEffort+=delta
    # print("controlEffort:",controlEffort)

    outflowAverage=np.mean(outflow)
    outflow-=outflowAverage
    outflow=outflow**2
    flashiness=np.sum(outflow)
    # print("flashiness:",flashiness)
    res=np.concatenate((state_list,pollution[:,np.newaxis]),axis=1)
    return np.array([overflow, peakOutflow, TSSload, controlEffort,flashiness]),res

if __name__=="__main__":
    data=[]
    data1=[]
    func_list = [rbc_easy, rbc_tss, rbc_depth, dqn]
    with Pool(4) as p:
        data_list = p.map(get_result, func_list)
    for d in data_list:
        data.append(d[0])
        data1.append(d[1])

    # data_easy,data_easy1=get_result(rbc_easy)
    # data.append(data_easy)
    # data1.append(data_easy1)
    # print('1')
    # data_tss,data_tss1=get_result(rbc_tss)
    # data.append(data_tss)
    # data1.append(data_tss1)
    # print('2')
    # data_depth,data_depth1=get_result(rbc_depth)
    # data1.append(data_depth1)
    # data.append(data_depth)
    # print('3')
    # data_dqn,data_dqn1=get_result(dqn)
    # data.append(data_dqn)
    # data1.append(data_dqn1)
    # data=np.array(data)
    # data1=np.array(data1)
    # print('4')

    labels = np.array(["Overflow", "Peak outflow", "TSS load", "Control effort", "Outflow flashiness"])
    dataLenth  = len(labels)      # 数据长度

    angles = np.linspace(0,2*np.pi,dataLenth,endpoint=False)   #根据数据长度平均分割圆周长

    #闭合
    data = np.array(data)
    data1 = np.array(data1)
    data = np.concatenate((data,data[:,0][:,np.newaxis]),axis=1)
    max1=np.max(data,axis=0)
    data=data/max1*100
    angles = np.concatenate((angles,[angles[0]]))
    labels=np.concatenate((labels,[labels[0]]))  #对labels进行封闭
    # print(data)
    
    filename="plot/"
    homedir = os.getcwd()#获取项目当前路径
    if not os.path.exists(homedir+'/'+filename):
        os.mkdir(homedir+'/'+filename)#创建pic文件夹，用于保存图片 
    plt.figure(facecolor="white")       #facecolor 设置框体的颜色
    plt.subplot(111,polar=True)     #将图分成1行1列，画出位置1的图；设置图形为极坐标图
    plt.plot(angles,data[0],'o-',linewidth=2,label = 'static')
    plt.fill(angles,data[0],alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
    plt.plot(angles,data[1],'o-',linewidth=2,label = 'tss')
    plt.fill(angles,data[1],alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
    plt.plot(angles,data[2],'o-',linewidth=2,label = 'depth')
    plt.fill(angles,data[2],alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
    plt.plot(angles,data[3],'o-',linewidth=2,label = 'dqn')
    plt.fill(angles,data[3],alpha=0.25)    #填充两条线之间的色彩，alpha为透明度
    plt.legend(loc=(0.98,0.1))
    plt.thetagrids(angles*180/np.pi,labels)          #做标签
    plt.figtext(0.52,0.95,'title',ha='center')   #添加雷达图标题
    plt.grid(True)
    plt.savefig(filename+"result.png")
    # plt.clf()

    plt.figure(figsize=(12,10))       #facecolor 设置框体的颜色
    plt.subplots_adjust(hspace=0.5)
    data1[:,:,5]/=10
    time_list = list(range(len(data1[0])))
    time_list =np.array(time_list)/12.0
    labels=["easy","rbc_tss","rbc_depth","dqn"]
    ylabels=["depth(m)","inflow(m3/s)","outflow(m3/s)","rainfall(mm/hr)","tss(mg/L)","OR","inflow tss(mg/L)","tss load(kg)"]
    titles=["depth","inflow","outflow","rainfall","tss","OR","inflow tss","tss load"]
    for i in range(8):
        plt.subplot(421+i)
        plt.plot(time_list,data1[0,:,i],label=labels[0])
        plt.plot(time_list,data1[1,:,i],label=labels[1])
        plt.plot(time_list,data1[2,:,i],label=labels[2])
        plt.plot(time_list,data1[3,:,i],label=labels[3])
        plt.xlabel("Time(h)")
        plt.ylabel(ylabels[i])
        
        plt.title(titles[i])
    # plt.legend()
    plt.savefig(filename+"allplot"+".png")
    plt.show()




