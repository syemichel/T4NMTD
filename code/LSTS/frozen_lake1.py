from contextlib import closing
from io import StringIO
from os import path
import numpy as np
from DFA import *
import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding
from gymnasium.envs.toy_text import FrozenLakeEnv
from typing import List, Optional, Dict, Any, Tuple

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
PICK = 4  # 新增：拿动作
DROP = 5  # 新增：扔动作

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFHF",
        "FFHFFFFP",
        "HFHFFHFF",
        "FFFPFFHF",
        "FFHFFFFH",
        "PHHFFGFF",
        "FHFFFFFH",
        "FFHFFHFF"
    ],
}

dfa_text = '''if(ds==@q1 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q1 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q1 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q1 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q1 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q3 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q3 ^ (p3 & ~p1 & ~p2 & ~p4)) then @q3
else if(ds==@q3 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q3 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q3 ^ (p4 & ~p1 & ~p2)) then @q6
else if(ds==@q3 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q4 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q4 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q4 ^ (p2 & ~p1 & ~p3 & ~p4)) then @q4
else if(ds==@q4 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q4 ^ (p4 & ~p1 & ~p3)) then @q6
else if(ds==@q4 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q2 ^ (true)) then @q2
else if(ds==@q5 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q5 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q5 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q5 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q5 ^ (p1 & ~p2 & ~p3 & ~p4)) then @q5
else if(ds==@q5 ^ (p4 & ~p2 & ~p3)) then @q6
else if(ds==@q6 ^ ((~p1 & ~p2) | (~p1 & ~p3) | (~p2 & ~p3))) then @q6
else if(ds==@q6 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2'''
# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False

def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]

class FrozenLakeEnv1(FrozenLakeEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.passenger_img = None

        # dfa and props setting
        self.dfa = DFATransformer(dfa_text)
        self.p1_loc = 15
        self.p2_loc = 27
        self.p3_loc = 40
        self.p4_loc = 45
        self.max_length = 200
        self.current_step = 0

        # DFA状态映射
        self.dfa_states = ['@q1', '@q2', '@q3', '@q4', '@q5', '@q6']
        self.dfa_state_to_idx = {state: idx for idx, state in enumerate(self.dfa_states)}

        # 新增：记录手上拿的是哪个passenger (0表示没有拿passenger, 1-3表示拿的passenger ID)
        self.carrying_passenger_id = 0

        # 新增：为每个passenger分配唯一ID和位置
        self.passengers = {}  # {passenger_id: (row, col)}
        self.passenger_locations = {}  # {(row, col): passenger_id or 0}

        # 初始化所有位置
        for row in range(nrow):
            for col in range(ncol):
                self.passenger_locations[(row, col)] = 0

        # 为地图上的passenger分配ID
        passenger_id = 1
        passenger_positions = []
        for row in range(nrow):
            for col in range(ncol):
                if desc[row, col] == b"P":
                    passenger_positions.append((row, col))

        # 确保有3个passenger，如果地图上不足3个，则在安全位置添加
        while len(passenger_positions) < 3:
            # 找一个安全的位置（不是洞、不是目标、不是起点）
            for row in range(nrow):
                for col in range(ncol):
                    if (desc[row, col] == b"F" and
                        (row, col) not in passenger_positions):
                        passenger_positions.append((row, col))
                        break
                if len(passenger_positions) >= 3:
                    break

        # 只取前3个位置
        passenger_positions = passenger_positions[:3]

        # 初始化passenger位置
        for i, (row, col) in enumerate(passenger_positions, 1):
            self.passengers[i] = (row, col)
            self.passenger_locations[(row, col)] = i

        # 确保有3个passenger，如果不足则设置默认位置
        for i in range(1, 4):
            if i not in self.passengers:
                # 设置一个默认位置（0,0）
                self.passengers[i] = (0, 0)

        nA = 6  # 修改：现在有6个动作
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            if action in [PICK, DROP]:
                # 拿和扔动作不改变位置
                newrow, newcol = row, col
            else:
                newrow, newcol = inc(row, col, action)

            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"H"
            reward = float(newletter == b"G")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(6):  # 修改：现在有6个动作
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if a in [PICK, DROP]:
                            # 拿和扔动作总是成功执行
                            li.append((1.0, *update_probability_matrix(row, col, a)))
                        elif is_slippery and a < 4:  # 只有移动动作受滑动影响
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        # 修改observation_space为Dict空间，支持stable-baselines3
        self.observation_space = spaces.Dict({
            "obs": Box(low=0, high=nS-1, shape=(1,), dtype=np.int64),
            "ds": Box(low=0, high=5, shape=(1,), dtype=np.int64),
            "carrying": Box(low=0, high=3, shape=(1,), dtype=np.int64),  # 0-3, 0表示没有拿，1-3表示拿的passenger ID
            "passenger_loc1": spaces.Box(low=0, high=max(nrow-1, ncol-1), shape=(2,), dtype=np.int32),
            "passenger_loc2": spaces.Box(low=0, high=max(nrow-1, ncol-1), shape=(2,), dtype=np.int32),
            "passenger_loc3": spaces.Box(low=0, high=max(nrow-1, ncol-1), shape=(2,), dtype=np.int32)
        })

        self.action_space = spaces.Discrete(nA)
        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.props = {'p1': False, 'p2': False, 'p3': False, 'p4': False}

    def _get_state_dict(self) -> Dict[str, Any]:
        """获取当前状态的字典表示"""
        return {
            "obs": np.int32(self.s),
            "ds": np.int32(self.dfa_state_to_idx[self.dfa.dfa_state]),
            "carrying": np.int32(self.carrying_passenger_id),
            "passenger_loc1": np.array(self.passengers[1], dtype=np.int32),
            "passenger_loc2": np.array(self.passengers[2], dtype=np.int32),
            "passenger_loc3": np.array(self.passengers[3], dtype=np.int32)
        }

    def step(self, a):
        current_row = self.s // self.ncol
        current_col = self.s % self.ncol

        # 处理拿和扔动作
        if a == PICK:
            # 拿动作：如果当前位置有passenger且手上没有passenger
            passenger_at_location = self.passenger_locations.get((current_row, current_col), 0)
            if passenger_at_location > 0 and self.carrying_passenger_id == 0:
                self.carrying_passenger_id = passenger_at_location
                self.passenger_locations[(current_row, current_col)] = 0
                # 更新passenger位置记录 - 设置为一个"无效"位置
                self.passengers[passenger_at_location] = (-1, -1)
            # 状态不变，没有移动
            self.lastaction = a
            terminated = False
            truncated = False
            reward = 0
        elif a == DROP:
            # 扔动作：如果手上有passenger，将其放到当前位置
            if self.carrying_passenger_id > 0:
                # 如果当前位置已经有passenger，不能放置
                if self.passenger_locations.get((current_row, current_col), 0) == 0:
                    passenger_id = self.carrying_passenger_id
                    self.passengers[passenger_id] = (current_row, current_col)
                    self.passenger_locations[(current_row, current_col)] = passenger_id
                    self.carrying_passenger_id = 0
            # 状态不变，没有移动
            self.lastaction = a
            terminated = False
            truncated = False
            reward = 0
        else:
            # 原有的移动动作
            transitions = self.P[self.s][a]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            p, s, r, t = transitions[i]
            self.s = s
            self.lastaction = a
            terminated = t
            reward = r

        # dfa step
        self.props = {'p1': False, 'p2': False, 'p3': False, 'p4': False}
        if self.carrying_passenger_id == 1:
            self.props['p1'] = True
        elif self.carrying_passenger_id == 2:
            self.props['p2'] = True
        elif self.carrying_passenger_id == 3:
            self.props['p3'] = True
        if self.s == self.p4_loc:
            self.props['p4'] = True

        terminated_dfa, if_success, if_failure = self.dfa.step(self.props)
        self.current_step += 1

        terminated = terminated or terminated_dfa
        if self.current_step >= self.max_length:
            truncated = True
        else:
            truncated = False

        if self.render_mode == "human":
            self.render()

        if terminated and not terminated_dfa:  # 原游戏结束条件
            self.dfa.dfa_state = '@q2'

        if self.dfa.dfa_state == '@q6':
            reward = 100.0
        elif terminated or truncated:
            reward = 0.0
        else:
            reward = 0

        '''if self.dfa.dfa_state == '@q4':
            reward = 100
            terminated = True
        elif terminated or truncated:
            reward = 0
        else:
            reward = -0.1'''

        # 返回新的状态格式
        state_dict = self._get_state_dict()
        info = {"prob": 1.0 if a in [PICK, DROP] else p}

        return (state_dict, reward, terminated, truncated, info)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.current_step = 0
        self.dfa.reset()

        # 重置passenger相关状态
        self.carrying_passenger_id = 0
        self.passengers = {}

        # 重新初始化所有位置
        for row in range(self.nrow):
            for col in range(self.ncol):
                self.passenger_locations[(row, col)] = 0

        # 重新为地图上的passenger分配ID
        passenger_id = 1
        passenger_positions = []
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.desc[row, col] == b"P":
                    passenger_positions.append((row, col))

        # 确保有3个passenger
        while len(passenger_positions) < 3:
            for row in range(self.nrow):
                for col in range(self.ncol):
                    if (self.desc[row, col] == b"F" and
                        (row, col) not in passenger_positions):
                        passenger_positions.append((row, col))
                        break
                if len(passenger_positions) >= 3:
                    break

        passenger_positions = passenger_positions[:3]

        # 初始化passenger位置
        for i, (row, col) in enumerate(passenger_positions, 1):
            self.passengers[i] = (row, col)
            self.passenger_locations[(row, col)] = i

        # 确保有3个passenger
        for i in range(1, 4):
            if i not in self.passengers:
                self.passengers[i] = (0, 0)

        if self.render_mode == "human":
            self.render()

        state_dict = self._get_state_dict()
        info = {"prob": 1}

        return state_dict, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake with Passengers")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/elf_left.png"),
                path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/elf_down.png"),
                path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/elf_right.png"),
                path.join(path.dirname(__file__), "/home/michel/anaconda3/envs/p2m/lib/python3.9/site-packages/gymnasium/envs/toy_text/img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"

        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)

                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                # 显示passenger（根据passenger_locations）
                passenger_id = self.passenger_locations.get((y, x), 0)
                if passenger_id > 0:
                    self.window_surface.blit(self.passenger_img, pos)

                    # 在passenger上显示ID号
                    font = pygame.font.Font(None, 24)
                    text = font.render(str(passenger_id), True, (255, 255, 255))
                    text_rect = text.get_rect(center=(pos[0] + self.cell_size[0]//2,
                                                    pos[1] + self.cell_size[1]//4))
                    self.window_surface.blit(text, text_rect)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None and self.lastaction < 4 else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

            # 如果手上有passenger，在elf旁边显示passenger图标和ID
            if self.carrying_passenger_id > 0:
                small_passenger = pygame.transform.scale(self.passenger_img,
                                                       (self.cell_size[0]//3, self.cell_size[1]//3))
                passenger_pos = (cell_rect[0] + self.cell_size[0] - self.cell_size[0]//3,
                               cell_rect[1])
                self.window_surface.blit(small_passenger, passenger_pos)

                # 显示手上passenger的ID
                font = pygame.font.Font(None, 16)
                text = font.render(str(self.carrying_passenger_id), True, (255, 0, 0))
                text_rect = text.get_rect(center=(passenger_pos[0] + self.cell_size[0]//6,
                                                passenger_pos[1] + self.cell_size[1]//6))
                self.window_surface.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]

        # 创建一个显示地图，显示当前passenger的实际位置和ID
        display_desc = [line[:] for line in desc]  # 深拷贝

        # 清除原始P标记
        for r in range(self.nrow):
            for c in range(self.ncol):
                if display_desc[r][c] == "P":
                    display_desc[r][c] = "F"

        # 根据当前passenger_locations重新标记，显示passenger ID
        for (r, c), passenger_id in self.passenger_locations.items():
            if passenger_id > 0 and display_desc[r][c] not in ["S", "G", "H"]:
                display_desc[r][c] = str(passenger_id)  # 显示passenger ID而不是P

        display_desc[row][col] = utils.colorize(display_desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            action_names = ['Left', 'Down', 'Right', 'Up', 'Pick', 'Drop']
            if self.lastaction < len(action_names):
                outfile.write(f"  ({action_names[self.lastaction]})\n")
            else:
                outfile.write("\n")
        else:
            outfile.write("\n")

        outfile.write("\n".join("".join(line) for line in display_desc) + "\n")

        # 显示当前状态信息
        state_dict = self._get_state_dict()
        outfile.write(f"State: obs={state_dict['obs']}, ds={self.dfa.dfa_state}, carrying={state_dict['carrying']}\n")
        outfile.write(f"Passengers: P1{tuple(state_dict['passenger_loc1'])}, P2{tuple(state_dict['passenger_loc2'])}, P3{tuple(state_dict['passenger_loc3'])}\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

# Elf and stool from https://franuka.itch.io/rpg-snow-tileset
# All other assets by Mel Tillery http://www.cyaneus.com/

if __name__ == '__main__':
    from stable_baselines3 import SAC
