import os
import sys
import gym
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
import torch.nn.init as init
import torch.nn.utils.prune as prune
import torch.backends.cudnn as cudnn

import VTT_RL

# ------------------------< SEED FIX >---------------------------------
custom_seed_val = 0
torch.manual_seed(custom_seed_val)
torch.cuda.manual_seed(custom_seed_val)
torch.cuda.manual_seed_all(custom_seed_val)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(custom_seed_val)

input_stack_size = 15

torch.cuda.memory_summary(device=None, abbreviated=False)

cur_path = os.path.dirname(__file__)
aim_agent_version = "agent_xyIncrease_cnn2d_v1.02.pth"

env = gym.make('VTT-v0').unwrapped


'''------------------< INPUT SHAPE DESCRIPTION >--------------------+
 |                                                                  |
 |inputs shape = [base part position, quertonion, vel, ang_vel (13)]|
 |              +[joint_state(12)]                                  |
 |              +[joint_vel(12)]                                    |    
 |              +[joint_reaction force(72)]                         |
 |              +[member pos, ori(42)]                              |
 |              +[action(12)]                 = 163                 | 
 +----------------------------------------------------------------'''

# inputs = 7 + 12 + 12 + 12 --old inputs 26016
inputs = 13+12+12+72+42+12

output = 12

max_episode_len = 2 ** 10
episode_persent_step = max_episode_len // 100

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(34980*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.to(device)
        x = self.bn1(self.conv1(x))
        x = x.view(-1, math.prod(x.shape))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)

        return out


BATCH_SIZE = 1  # 256 64 1
GAMMA = 0.999
EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 100
LEARNING_RATE = 0.008  # 0.001 0.01

env.reset()

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(env.action_space.n * output).to(device)  # cnn2d prune
target_net = DQN(env.action_space.n * output).to(device)  # cnn2d prune


# Global prune
parameters_to_prune = (
    (policy_net.conv1, 'weight'),
    (policy_net.fc1,   'weight'),
    (policy_net.fc2,   'weight'),
    (policy_net.fc3,   'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

parameters_to_prune = (
    (target_net.conv1, 'weight'),
    (target_net.fc1,   'weight'),
    (target_net.fc2,   'weight'),
    (target_net.fc3,   'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

total1   = sum(p.numel() for p in policy_net.parameters())
trainpa1 = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
total2   = sum(p.numel() for p in target_net.parameters())
trainpa2 = sum(p.numel() for p in target_net.parameters() if p.requires_grad)

print("Params of policy net:", total1,
      "Trainable params of policy net:", trainpa1)
print("Params of target net:", total2,
      "Trainable params of target net:", trainpa2)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
# criterion = torch.jit.script(nn.MSELoss())
criterion = torch.jit.script(nn.SmoothL1Loss())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):  # steps_done,
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-0.1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # return policy_net(state).max(1)[1].view(1, 1)
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).view(output, n_actions).max(-1)[1]
    else:
        # return torch.randn([output], device=device, dtype=torch.float32)
        return torch.tensor([random.randrange(n_actions) for _ in range(output)], device=device, dtype=torch.float32)


reward_hisroty = []
loss_hisroty = []

fig = plt.figure(1)
plt.clf()

ax_one = plt.gca()
ax_loss = ax_one.twinx()

plt.title('Training...')
ax_one.set_xlabel("Episode")
ax_one.set_ylabel("Reward_avg/life")
ax_loss.set_ylabel("Loss_avg/life")

p_reward, = ax_one.plot([], color='deepskyblue', label="Reward_avg/life")
p_mean, = ax_one.plot([], color='navy', label="Reward_filter_100")
p_loss, = ax_loss.plot([], color='tomato', label="Loss_avg/life")

ax_one.grid(True)
ax_loss.legend(handles=[p_reward, p_mean, p_loss], loc='upper left')


def plot_durations():
    objective_val = torch.tensor(reward_hisroty, dtype=torch.float)

    ax_one.plot(objective_val.numpy(), color='deepskyblue')

    # Take 100 episode averages and plot them too
    if len(objective_val) >= 100:
        means = objective_val.unfold(0, 100, 1).mean(1).view(-1)
        loss_val = torch.tensor(loss_hisroty, dtype=torch.float)
        means = torch.cat((torch.zeros(99), means))
        ax_one.plot(means.numpy(), color='navy')

    if len(loss_hisroty) > 0:
        loss_val = torch.tensor(loss_hisroty, dtype=torch.float)
        ax_loss.plot(loss_val.numpy(), color='tomato')

    plt.pause(0.001)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    start = time.time()
    batch = Transition(*zip(*transitions))
    print("\tbatch spread   : \t", time.time() - start)

    start = time.time()
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch  = torch.cat(batch.state )
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    print("\tnon_final_mask : \t", time.time() - start)

    start = time.time()
    state_thr_policy_net = policy_net(
        state_batch).view(BATCH_SIZE, output, n_actions)
    state_action_values = state_thr_policy_net.gather(
        2, action_batch.view(BATCH_SIZE, output, 1).to(dtype=torch.int64))
    print("\tstate_action   : \t", time.time() - start)

    start = time.time()
    next_state_values = torch.zeros((BATCH_SIZE, output), device=device)
    next_state_thr_policy_net = target_net(
        non_final_next_states).view(BATCH_SIZE, output, n_actions)
    next_state_values[non_final_mask, :] = next_state_thr_policy_net.max(2)[
        0].detach()
    # Compute the expected Q values
    expected_state_action_values = (
        next_state_values * GAMMA) + reward_batch.unsqueeze(1).repeat(1, output)
    print("\texpected value : \t", time.time() - start)
    
    start = time.time()
    # Compute Huber loss
    loss = criterion(state_action_values.view(1, -1).float(),
                     expected_state_action_values.view(1, -1).float())
    print("\tloss calculation: \t", time.time() - start)
    
    start = time.time()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
        # print(param.grad)
    print("\tzero grad & clamp: \t", time.time() - start)
    
    start = time.time()
    optimizer.step()
    print("\toptimizer step   : \t", time.time() - start)

    return loss

def potter(input, seed_tensor):
    randomized_input = torch.empty(size=seed_tensor.shape)
    for i, index in zip(range(seed_tensor.shape[0]),seed_tensor):
        randomized_input[i,:] = input[index]

    return randomized_input

def save_model(path):
    torch.save({
        'policy': policy_net.state_dict(),
        'target': target_net.state_dict(),
        'total_reward': reward_hisroty,
        'loss_hisroty': loss_hisroty,
    }, path)


def load_model(path):
    checkpoint = torch.load(path)
    policy_net.load_state_dict(checkpoint['policy'])
    target_net.load_state_dict(checkpoint['target'])
    reward_hisroty = checkpoint['total_reward']
    loss_hisroty = checkpoint['loss_hisroty']

    return reward_hisroty, loss_hisroty


if __name__ == '__main__':
    total_reward = []
    total_loss = []
    num_episodes = int(sys.maxsize / 2)

    potter_indexies = []
    for i in range(custom_seed_val, custom_seed_val+input_stack_size):
        random.seed(i)
        potter_indexies.append(random.sample(range(inputs),inputs))
    potter_indexies = torch.tensor(potter_indexies, device= 'cuda')

    # model_path = "/home/elitedog/vttopt/VTT_pybullet_DQN/_agent_data/" + aim_agent_version        #Jeff's
    # model_path = "/home/sjryu/PythonEX/VTT_pybullet_DQN_model_compressed/_agent_data/" + aim_agent_version  # SJ's
    model_path = cur_path + '/_agent_data/' + aim_agent_version

    if os.path.exists(model_path):
        reward_hisroty, loss_hisroty = load_model(model_path)

    for i_episode in range(num_episodes):

        # Initialize the environment and state
        state = torch.tensor(env.reset(), device= device)
        state = potter(state, potter_indexies)
        state = state.view(1, 1, state.shape[0], state.shape[1])

        for t in count():

            # Select and perform an action
            start = time.time()
            action = select_action(state)
            
            print("action select  : \t", time.time() - start)
            start = time.time()

            next_state, reward, done, _ = env.step(action)

            print("step time      : \t", time.time() - start)
            start = time.time()

            # survive_reward = math.exp(-0.0001*t)
            survive_reward = 0.0
            reward = torch.tensor([reward + survive_reward], device=device)
            total_reward.append(reward)

            # Move to the next state
            next_state = torch.tensor(next_state, device= device)
            next_state = potter(next_state, potter_indexies)
            next_state = next_state.view(1, 1, next_state.shape[0], next_state.shape[1])

            # Add tuple to memory
            memory.push(state, action, next_state, reward)

            # storage next_state as state
            state = next_state

            
            start = time.time()
            # Perform one step of the optimization (on the policy network)
            loss = optimize_model()
            print("optimize_model : \t", time.time() - start)

            if loss != None:
                total_loss.append(loss)

            if done or t > max_episode_len:
                reward_hisroty.append(sum(total_reward) / t)
                if (len(total_loss) != 0) and (len(loss_hisroty) != 0):
                    loss_hisroty.append(sum(total_loss) / len(total_loss))
                else:
                    loss_hisroty.append(0)
                total_reward = []
                total_loss = []
                time.sleep(0.01)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            save_model(model_path)

            steps_done = 0

        if i_episode % 1 == 0:  # (TARGET_UPDATE*10)
            plot_durations()

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
