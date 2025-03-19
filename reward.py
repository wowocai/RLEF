import math
def calculate_cost(state):
    # size =[state[10][1],state[10][2]]

    dtype = state[0][5] * 2.5 #病害类型，从normalization复原 #2.5
    origin_size = state[0][6] * 18.356#病害尺寸，复原normalization
    if abs(dtype - 1) < 0.0001:
        size  = [round(origin_size/0.05)+1, origin_size*0.5]
    else:
        size = [round(origin_size*3)+1,origin_size]
    process_date = 50
    for i in range(1,50):
        if state[i][6] - 0.0 < 0.00001:
            process_date = i

         
    # print(action,response,dtype)
    AC13_cost = 300
    AC25_cost = 250
    pour_cost = 3
    seal_cost = 7
    manpower = 2
    # if dtype < 0.4 or (dtype>=0.4 and action < 1.1):
    if process_date < 50: #method != 0
        action = state[-1][7]#第8个特征，养护方法
        if abs(action - 1.2) < 0.001:#'灌缝' :      
            # print(1)
            return size[0] * (seal_cost + manpower)
        elif abs(action - 1.1) < 0.001:#'贴缝':
            # print(2)
            return size[0] * (pour_cost + manpower)
        elif abs(action - 1) < 0.001:#'AC-13沥青补坑':
            # print(3)
            return (size[1]**0.5 + 0.5)**2 * (AC13_cost + manpower)
        elif abs(action - 1.05) < 0.001:#'AC-13沥青补坑,贴缝':
            # print(4)
            return (size[1]**0.5 + 0.5)**2 * (AC13_cost + manpower) + size[0] * (seal_cost + manpower)
        elif abs(action - 0.9) < 0.001:#'AC-13沥青补坑,灌缝':
            # print(5)
            return (size[1]**0.5 + 0.5)**2 * (AC13_cost + manpower)  + size[0] * (pour_cost + manpower)
        elif abs(action - 0.85) < 0.001:#'AC-25沥青补坑,贴缝':
            # print(6)
            return (size[1]**0.5 + 0.5)**2 * (AC25_cost + manpower) + size[0] * (seal_cost + manpower)
        elif abs(action - 0.8) < 0.001:#'AC-25沥青补坑':
            # print(7)
            return (size[1]**0.5 + 0.5)**2 * (AC25_cost + manpower)
        else:
            # print(8)
            return 0
    else:
        # print(0)
        return 0

def calculate_performance(state):
    perform = state[:,4]
    w_t = 3.9
    w_m = 670
    traffic_punish = (sum(perform)-perform[0]*50)
    maintain_punish = perform[-1] -perform[0]
    return  traffic_punish*w_t + maintain_punish*w_m

def calculate_reward(state):
    # Define weights to balance cost and pavement performance
    cost_weight = 0.4
    performance_weight = 0.55

    # Calculate the cost component of the reward
    cost = calculate_cost(state)  # Your implementation for calculating the cost goes here

    # Calculate the pavement performance component of the reward
    performance = calculate_performance(state)  # Your implementation for calculating pavement performance goes here

    # punish wrong method for specific type
    if 1.01 < state[0][5] * 2.5  and state[-1][7] > 1:
        punish = 500
    elif state[0][5] * 2.5 <= 1.01 and (state[-1][7] - 1 <0.001 or state[-1][7] - 0.8 <0.001):
        punish = 200
    else:
        punish = 0
    if state[0][5]-0 < 0.001 and state[-1][7] - 0 > 0.001:#no distress but maintain
        punish += 500
    if state[0][5]-0 > 0.001 and state[-1][7] - 0 < 0.001:# distress but no maintain
        
        punish += 100
    # Combine the cost and performance components using the defined weights
    # print(cost, performance)
    reward = cost_weight * (-cost) - performance_weight * performance - punish
    # print(cost,performance)
    # print(reward)

    return reward

def calculate_performance_data(state):
    perform = state[:,4]
    traffic_punish = (sum(perform)-perform[0]*50)
    maintain_punish = perform[-1] -perform[0]
   
    return  traffic_punish, maintain_punish