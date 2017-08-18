# Thompson Sampling
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# Importing Dataset
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')
X=dataset.iloc[:,[3,4]].values

# Implementing Thompson Sampling
N=10000
d=10
ads_selected=[]
num_of_rewards_1=[0]*d
num_of_rewards_0=[0]*d
total_reward = 0
for n in range(0,N):
    ad=0
    max_random =0
    for i in range (0,d):
        random_beta = random.betavariate(num_of_rewards_1[i]+1,num_of_rewards_0[i]+1)
        
        if random_beta > max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n , ad]
    if reward ==1:
        num_of_rewards_1[ad]=num_of_rewards_1[ad]+1
    else:
        num_of_rewards_0[ad]=num_of_rewards_0[ad]+1
    total_reward=total_reward+reward
    

# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ad selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad clicked')