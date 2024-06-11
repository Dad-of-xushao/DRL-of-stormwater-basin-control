from pyswmm import Simulation, Nodes,Links,RainGages
import os
import numpy as np
import matplotlib.pyplot as plt

def feet_to_m(feet):
    return feet*0.3048
def m_to_feet(m):
    return m/0.3048
def feet_to_mm(feet):
    return feet*304.8
def mm_to_feet(feet):
    return feet/304.8
def cfs_to_m3s(cfs):
    return cfs*0.028316846592 
class swmmEnv():
    def __init__(self,inp_file,T=None) -> None:
        self.input_file = inp_file
        self.len=len(self.input_file)
        input_file=self.input_file[np.random.randint(self.len)]
        self.sim = Simulation(input_file)  # read input file

        self.control_time_step = 300  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step

        node_object = Nodes(self.sim)  # init node object
        self.Stup = node_object["93-49743"]
        self.St1 = node_object["93-50074"]
        self.St1.initial_depth=0
        # self.St1.full_depth=4.5

        link_object = Links(self.sim)  # init link object
        self.ORup = link_object["OR39"]
        self.OR = link_object["OR38"]
        self.conduitUp=link_object['C08']
        self.conduitDown=link_object['C95-53307']

        rain_object=RainGages(self.sim)# init RainGages object
        self.rain=rain_object['R9162']
        self.sim.start()

        sim_len = self.sim.end_time - self.sim.start_time
        if T:
            self.T=T
        else:
            self.T = int(sim_len.total_seconds()/self.control_time_step)
        # print(sim_len.total_seconds())
        self.t = 1
        

        self.OR.target_setting = 0

    def step(self, action):
        # self.ORup.target_setting=0
        self.OR.target_setting = action/10.0
        last_depth=feet_to_m(self.St1.depth)
        # self.rain.total_precip = 0
        self.sim.__next__()
        
        self.t+=1
        current_depth=feet_to_m(self.St1.depth)
        current_rainfall=feet_to_mm(self.rain.rainfall/12)
        up_flow=cfs_to_m3s(self.conduitUp.flow)
        down_flow=cfs_to_m3s(self.conduitDown.flow)
        in_folw=cfs_to_m3s(self.St1.total_inflow)
        out_flow=cfs_to_m3s(self.St1.total_outflow)
        tss_concentration= self.St1.pollut_quality['TSS']
        inflow_quality=self.conduitUp.pollut_quality['TSS']
        next_state=[current_depth,in_folw,out_flow,current_rainfall,tss_concentration,action,inflow_quality]

        reward=0
        reward-=100*(action-self.lastAction)*(action-self.lastAction)
        if out_flow>self.maxOutflow:
            reward-=1
        if current_depth>0:
            reward-=200*(current_depth-4.18)
        elif current_depth-last_depth<0:
            reward+=15
        else:
            reward-=15
        if tss_concentration>10:
            reward-=out_flow*tss_concentration*50
        else:
            reward-=out_flow*tss_concentration*20
        # elif tss_concentration<10:
        #     reward+=50000*out_flow
        
        self.lastAction=action
        done=False
        if self.t < self.T:
            done = False
        else:
            done = True

        return next_state, reward, done
    
    def reset(self):
        self.sim.close()
        input_file=self.input_file[np.random.randint(self.len)]
        self.sim = Simulation(input_file)  # read input file
        self.control_time_step = 300  # control time step in seconds
        self.sim.step_advance(self.control_time_step)  # set control time step

        node_object = Nodes(self.sim)  # init node object
        self.St1 = node_object["93-50074"]
        self.St1.initial_depth=m_to_feet(0)
        self.St1.full_depth=m_to_feet(2.5)
        # print(self.St1.full_depth)
        # self.St1.surcharge_depth=0

        link_object = Links(self.sim)  # init link object
        self.OR = link_object["OR38"]
        self.conduitUp=link_object['C08']
        self.conduitDown=link_object['C95-53307']

        rain_object=RainGages(self.sim)# init RainGages object

        self.rain=rain_object['R9162']
        self.sim.start()

        self.t=1
        current_depth=feet_to_m(self.St1.depth)
        current_rainfall=feet_to_mm(self.rain.rainfall/12)
        up_flow=cfs_to_m3s(self.conduitUp.flow)
        down_flow=cfs_to_m3s(self.conduitDown.flow)
        in_folw=cfs_to_m3s(self.St1.total_inflow)
        out_flow=cfs_to_m3s(self.St1.total_outflow)
        tss_concentration= self.St1.pollut_quality['TSS']
        inflow_quality=self.conduitUp.pollut_quality['TSS']
        state=[current_depth,in_folw,out_flow,current_rainfall,tss_concentration,0,inflow_quality]
        self.maxOutflow=0
        self.lastAction=0
        return state
    
    def close(self):
        # self.sim.report()
        self.sim.close()
    
if __name__=="__main__":
    swmm=swmmEnv("model/runfile.inp")
    
    done=False
    state_list=[]
    swmm.reset()
    while not done:
        next_state, reward, done=swmm.step(0)
        state_list.append(next_state)
    time_list = list(range(len(state_list)))
    time_list =np.array(time_list)/12.0
    state_list=np.array(state_list)
    swmm.close()
    plt.plot(time_list,state_list[:,0])
    plt.xlabel("Time(h)")
    plt.ylabel("depth(m)")
    plt.title("depth")
    plt.savefig("rbc/depth.png")
    plt.cla()

    plt.plot(time_list,state_list[:,1],label="inflow")
    plt.title("inflow")
    plt.xlabel("Time(h)")
    plt.ylabel("inflow(m3/s)")
    plt.savefig("rbc/inflow.png")
    plt.cla()
    
    plt.plot(time_list,state_list[:,2],label="outflow")
    plt.title("outflow")
    plt.xlabel("Time(h)")
    plt.ylabel("outflow(m3/s)")
    plt.savefig("rbc/outflow.png")
    plt.cla()
    
    plt.plot(time_list,state_list[:,3])
    plt.title("rainfall")
    plt.xlabel("Time(h)")
    plt.ylabel("rainfall(mm/hr)")
    plt.savefig("rbc/rainfall.png")
    plt.cla()
    
    plt.plot(time_list,state_list[:,4])
    plt.title("tss")
    plt.xlabel("Time(h)")
    plt.ylabel("tss(mg/L)")
    plt.savefig("rbc/tss.png")
    plt.cla()
    
    pollution=state_list[:,2]*state_list[:,4]
    len=len(pollution)
    for i in range(1,len):
        pollution[i]+=pollution[i-1]
    plt.plot(time_list,pollution)
    plt.title("pollution")
    plt.xlabel("Time(h)")
    plt.ylabel("pollution(g)")
    plt.savefig("rbc/pollution.png")
    plt.cla()