from statistics import median

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import seaborn as sns
import matplotlib.pyplot as plt
from torch.onnx.symbolic_opset9 import tensor
import pickle
import looker
from itertools import count
import solver

Transition = namedtuple('Transition',
                        ('prev_action','state', 'action', 'next_state', 'reward'))
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


grid=looker.get_map(1)
savefile="9x9_2.pth"
savememory="9x9memory_2.pkl"
savegoodmemory="9x9goodmemory_2.pkl"
def save():
    # Сохранение
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': EPS,
    }, savefile)

    with open(savememory, "wb") as f:
        pickle.dump(memory, f)

    with open(savegoodmemory, "wb") as f:
        pickle.dump(goodmemory, f)
def load_memory():
    try:
        with open(savememory, "rb") as f:
            memory = pickle.load(f)
            return memory
    except FileNotFoundError:
        return ReplayMemory(10000)

def load_good_memory():
    try:
        with open(savegoodmemory, "rb") as f:
            memory = pickle.load(f)
            return memory
    except FileNotFoundError:
        return ReplayMemory(10000)

class DQN(nn.Module):

    def __init__(self, size_x, size_y):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))

        )
        self.flatten = nn.Flatten()
        fc_size = 256 * 4 * 4

        # Advantage stream
        self.adv = nn.Sequential(
            nn.Linear(fc_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * size_x * size_y)
        )

        # Value stream
        self.val = nn.Sequential(
            nn.Linear(fc_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - torch.mean(adv,dim=1, keepdim=True)

policy_net=DQN(grid.xsize,grid.ysize).to(device)
target_net=DQN(grid.xsize,grid.ysize).to(device)
target_net.load_state_dict(policy_net.state_dict())

BATCH_SIZE = 128
GAMMA = 0.99
EPS = 0.9
EPS_END = 0.05
EPS_DECAY = 0.9995
TAU = 0.005
LR = 1e-4
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory=load_memory()
goodmemory=load_good_memory()

try:
    checkpoint = torch.load(savefile, map_location=device)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    EPS = checkpoint['epsilon']
    EPS = 0.5
except FileNotFoundError:
    save()
 # не забыть вернуть в режим тренировки


def select_action(state, epsilon):
    # try:
    #     if random.random()<0.3:
    #         return solver.get_route_solver(grid)[-1]
    # except Exception:
    #     pass
    if random.random() < epsilon:
        return grid.get_random_action()  # Explore
    else:
        if state.dim() == 3:
            state = state.unsqueeze(0)

        q_values = policy_net(state)
        return torch.argmax(q_values).item()  # Exploit
def sample_mixed_batch(batch_size):
    good_size =int(batch_size*0.75)
    normal_size = batch_size - good_size

    good_samples = goodmemory.sample(min(good_size, len(goodmemory)))
    normal_samples = memory.sample(min(normal_size, len(memory)))

    return good_samples + normal_samples

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = sample_mixed_batch(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    try:
        state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    except RuntimeError:
        print(policy_net(state_batch).shape)
        print(action_batch.shape)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 5000
else:
    num_episodes = 50

rewards=[]
steps=[]
res=[]
flags=[]

for sol in range(50):
    print(f"Learning number {sol}")
    grid = looker.get_random_map("small", True)
    done = 0
    state = grid.reset()
    state = torch.tensor(state.detach().clone(), dtype=torch.float32, device=device).unsqueeze(0).to(device)
    prev_action=-1
    while done == 0:
        actions = solver.get_route_solver(grid)

        if not actions:
            break

        for action in actions:
            observation, reward, done = grid.step(action)

            if done == 1 or done == -1 or done == -2:
                next_state = None
            else:
                next_state = torch.tensor(observation.detach().clone(), dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(torch.tensor([prev_action], dtype=torch.long, device=device),state, torch.tensor([action], dtype=torch.long, device=device), next_state, reward)
            goodmemory.push(torch.tensor([prev_action], dtype=torch.long, device=device), state,
                        torch.tensor([action], dtype=torch.long, device=device), next_state, reward)
            prev_action=action

            state = next_state
            if done == 1 or done == -1 or done == -2:
                break


            optimize_model()


try:
    ref=[]
    for i_episode in count():
        acts=[]
        # Initialize the environment and get its state

        if i_episode==0:
            ref=[]
            grid=looker.get_random_map("small",True)
            done=0
            while done == 0:
                actions = solver.get_route_solver(grid)

                if not actions:
                    break

                for action in actions:
                    observation, reward, done = grid.step(action)
                    ref.append(action)
                    if done == 1 or done == -1 or done == -2:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation.detach().clone(), dtype=torch.float32,
                                                  device=device).unsqueeze(0)

                    if done == 1 or done == -1 or done == -2:
                        break
        state = grid.reset()

        state = torch.tensor(state.detach().clone(), dtype=torch.float32, device=device).unsqueeze(0).to(device)

        rew=0

        if i_episode%50==0:
            save()

        c=0
        f=0
        prev_action=-1
        while True:

            action = select_action(state,EPS)
            # if i_episode < 200 and not(grid.area[grid.startx][grid.starty].opn):
            #     action = grid.startx*grid.ysize+grid.starty
            observation, reward, done = grid.step(action)

            if action>grid.ysize*grid.xsize:
                f+=1




            acts.append(action)
            if len(acts)>=3 and acts[-1]==acts[-2] and acts[-2]==acts[-3]:
                i=-1
                while abs(i)<len(acts) and acts[i]==acts[-1]:
                    i-=1
                reward -= abs(i)*10

                # if abs(i)>5:
                #     reward=-10
                #     done = -2
            rew += reward
            if ref.count(action) and acts.count(action)<=ref.count(action):
                reward+=100
            else:
                reward-=10
            reward = torch.tensor([reward], device=device)


            if done==1 or done == -1 or done ==-2:
                next_state = None
            else:
                next_state = torch.tensor(observation.detach().clone(), dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(torch.tensor([prev_action], dtype=torch.long, device=device),state, torch.tensor([action], dtype=torch.long, device=device), next_state, reward)
            if reward>0 or done==1:
                goodmemory.push(torch.tensor([prev_action], dtype=torch.long, device=device), state,
                            torch.tensor([action], dtype=torch.long, device=device), next_state, reward)
            prev_action=action
            # Move to the next state
            state = next_state
            if True:

            # Perform one step of the optimization (on the policy network)

                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            c+=1
            if done==1:
                print(f"HOLY HELL Episode {i_episode} won with reward {rew} {EPS}" )
                print(acts)
                rewards.append(rew)
                steps.append(len(acts))
                res.append(done)
                flags.append(f)
                break
            elif done == -1 or done == -2:
                print(f"Episode {i_episode} ended with {done} and reward {rew} {EPS}" )
                print(acts)
                rewards.append(rew)
                steps.append(len(acts))
                res.append(done)
                flags.append(f)
                break
            else:
                next_state = torch.tensor(observation.detach().clone(), dtype=torch.float32, device=device).unsqueeze(0)
        if True:
            EPS = max(EPS_END, EPS * EPS_DECAY)
except KeyboardInterrupt:


    plt.figure()
    lg1=sns.lineplot(data=rewards)
    plt.xlabel("Эпизоды")
    plt.ylabel("Награды")

    mdrew=[]
    for i in range(0,len(rewards),50):
        mdrew.append(median(rewards[i:min(len(rewards)-1,i+50)]))

    plt.figure()
    lgmd=sns.lineplot(data=mdrew)
    plt.xlabel("Эпизоды")
    plt.ylabel("Средние награды")

    plt.figure()
    lg2 = sns.lineplot(data=steps, color='red')
    plt.xlabel("Эпизоды")
    plt.ylabel("Шаги")

    plt.figure()
    lg3 = sns.lineplot(data=res, color='green')
    plt.xlabel("Эпизоды")
    plt.ylabel("Результат")

    plt.figure()
    lg4 = sns.lineplot(data=flags, color='yellow')
    plt.xlabel("Эпизоды")
    plt.ylabel("Флаги")

    plt.show()


