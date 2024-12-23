import torch
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from game_net import GameNet
import os

JPG_DIR = 'jpg'
WEIGHTS_DIR = 'weights'

os.makedirs(JPG_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

#Вероятность случайного хода
EPSILON = 0.1

class Stone:
    def __init__(self, id: int, position: Tuple[int, int]):
        self.id = id
        self.position = position

class Agent:
    def __init__(self, id: int, position: Tuple[int, int], net=None):
        self.id = id
        self.position = position
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) if net is not None else None
        self.memory = None
        self.next_move: Tuple[int, int] = None
        self.field_size = None
        self.agents_num = None
        self.predictions = None
        self.move_raw = None
        self.loss_func = torch.nn.MSELoss()

    def init_memory(self, current_state, memory_len=5):
        d, w, h = current_state.shape
        self.memory = torch.cat([current_state] * memory_len, dim=0)
        self.field_size = (w, h)
        self.agents_num = d - 1

    #Debug
    def _choose_random_point(self):
        variants = torch.argwhere(self.memory[-1-self.agents_num] == 1)
        return random.choice(variants).tolist()

    def choose_move(self):
        if self.net is not None:
            self.predictions = self.net(self.memory)
            self.move_raw = torch.argmax(self.predictions).unsqueeze(0)
            if (random.random() < EPSILON):
                self.move_raw = torch.tensor([random.randint(0,35)])
            self.next_move = (self.move_raw.item() // 6, self.move_raw.item() % 6)
        else:
            self.next_move = self._choose_random_point()

    def _calculate_reward(self, old_state, new_state):
        old_stones_map = old_state[0]
        new_stones_map = new_state[0]
        #Сходил на пустую клетку
        if old_stones_map[self.position[0], self.position[1]] == 0:
            reward = -10
        #Сходил к камню, но он не был убран
        elif new_stones_map[self.position[0], self.position[1]] == 1:
            reward = 10
        #Сходил к камню, камень убран. Штрафует если у одного камня больше двоих агентов. 2 агента = +50, 5 агентов = +10
        else:
            players_on_this_cell = 0
            for i in range(1,6):
                if new_state[i, self.position[0], self.position[1]] == 1:
                    players_on_this_cell += 1
            reward = 50 - (40 * (players_on_this_cell - 2) / 3)
        return reward

    def _update_memory(self, new_state):
        self.memory = torch.cat([self.memory[6:], new_state], dim=0)

    def learn(self, new_state):
        if self.net is not None:
            old_state = self.memory[24:]
            reward = self._calculate_reward(old_state, new_state)
            self._update_memory(new_state)
            next_predictions = self.net(self.memory)
            Q_predicted = torch.max(self.predictions)
            Q_next = torch.max(next_predictions)
            Q_target = reward + Q_next

            self.optimizer.zero_grad()
            loss = self.loss_func(Q_target, Q_predicted)
            loss.backward()
            # Нужно для GameNetv1, GameNetv2
            if (self.net.fc1.weight.grad.norm() < 0.0001):
                self.net.fc1.weight.grad.data += torch.FloatTensor([0.001])
            self.optimizer.step()


class Field:
    def __init__(self, size: Tuple[int,int], agents: int, stones: int, models: list = None, save_every_move=False, dummy=False):
        self.size = size
        self.agents, self.stones = self._distribute_players(agents, stones, models, dummy)
        self.move_number = 0
        self.save_every_move = save_every_move
        self.current_state = None
        self._set_current_state()
        for agent in self.agents:
            agent.init_memory(current_state=self.current_state)

        if self.save_every_move: self.save_jpg(filename=f'{JPG_DIR}/{self.move_number}.png')

    def _distribute_players(self, agents, stones, models, dummy=False):
        random_matrix = np.random.permutation(np.arange(0, self.size[0]*self.size[1])).reshape(self.size[0], self.size[1])
        ri = 0
        agents_list = []
        stones_list = []
        for i in range(1,agents+1):
            x, y = np.argwhere(random_matrix == ri)[0]
            if dummy:
                net = None
            elif models is not None:
                net = models.pop()
            else:
                net = GameNet()
            agent = Agent(id=i, position=(x,y), net=net)
            agents_list.append(agent)
            ri += 1
        for i in range(1,stones+1):
            x, y = np.argwhere(random_matrix == ri)[0]
            stone = Stone(id=i, position=(x, y))
            stones_list.append(stone)
            ri += 1
        return agents_list, stones_list

    def _set_current_state(self):
        current_state = torch.zeros((len(self.agents)+1, self.size[0], self.size[1]))
        for stone in self.stones:
            current_state[0, stone.position[0], stone.position[1]] = 1
        agent_number = 1
        for agent in self.agents:
            current_state[agent_number, agent.position[0], agent.position[1]] = 1
            agent_number += 1
        self.current_state = current_state

    def _create_agent_map(self):
        agents_map = [[[] for _ in range(self.size[0])] for _ in range(self.size[1])]
        for agent in self.agents:
            x, y = agent.position
            agents_map[x][y].append(agent.id)
        return agents_map

    def _plan_move(self):
        for agent in self.agents:
            agent.choose_move()

    def _learn(self):
        for agent in self.agents:
            agent.learn(self.current_state)

    def make_move(self):
        self._plan_move()

        for agent in self.agents:
            agent.position = agent.next_move
        agents_map = self._create_agent_map()
        self.stones = [stone for stone in self.stones if len(agents_map[stone.position[0]][stone.position[1]]) < 2]

        self._set_current_state()
        self._learn()

        self.move_number += 1

        if self.save_every_move: self.save_jpg(filename=f'{JPG_DIR}/{self.move_number}.png')

        return len(self.stones)

    def play(self):
        move = 0
        print(f'Move ', end='')
        stones = -1
        while stones != 0:
            stones = self.make_move()
            move += 1
            if move % 100 == 0:
                print(f'{move} ', end='')
        print('')
        models = []
        for agent in self.agents:
            models.append(agent.net)
        return self.move_number, models



    def save_jpg(self, filename="field_state.png"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        ax.set_xticks(range(self.size[0]))
        ax.set_yticks(range(self.size[1]))
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=1)

        for stone in self.stones:
            x, y = stone.position
            circle = patches.Circle((x + 0.5, y + 0.5), 0.3, color='black')
            ax.add_patch(circle)

        agents_map = self._create_agent_map()
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if agents_map[x][y] != []:
                    text = " ".join(map(str, agents_map[x][y]))
                    ax.text(x + 0.1, y + 0.8, text, fontsize=12, color='red', fontweight='bold')

        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.title("Field State")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    models = None
    moves_list = []
    for i in range(1000):
        if i==999:
            field = Field((6, 6), 5, 10, models=models, save_every_move=True, dummy=False)
        else:
            field = Field((6, 6), 5, 10, models=models, save_every_move=False, dummy=False)
        # field = Field((6, 6), 5, 10, models=models, save_every_move=True, dummy=False)
        moves, models = field.play()
        print(f'Game {i + 1}/1000 \t moves: {moves}')
        moves_list.append(moves)
    plt.plot([i+1 for i in range(len(moves_list))], moves_list)
    plt.xlabel('Игра')  # Подпись для оси х
    plt.ylabel('Количество ходов до победы')  # Подпись для оси y
    plt.title('GameNetv3, +50 уборка, +10 ход к камню, -10 ход в пустую клетку')  # Название
    plt.show()

    for i in range(len(models)):
        model = models[i]
        torch.save(model.state_dict(), f'{WEIGHTS_DIR}/{i+1}.pt')
