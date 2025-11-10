from __future__ import annotations

import copy
import sys
import time

import ray
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Door, Goal,  Ball
from util.DFA import *
from util.wrapper import *
from typing import Any, Iterable, SupportsFloat, TypeVar
import numpy as np
from gymnasium.core import ActType, ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace

dfa_text = '''if(ds==@q1 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q1 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q1 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q1 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q1 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q3 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q3 ^ (p3 & ~p1 & ~p2 & ~p4)) then @q3
else if(ds==@q3 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q3 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q3 ^ (p3 & p4 & ~p1 & ~p2)) then @q6
else if(ds==@q3 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q4 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q4 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q4 ^ (p2 & ~p1 & ~p3 & ~p4)) then @q4
else if(ds==@q4 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q4 ^ (p2 & p4 & ~p1 & ~p3)) then @q6
else if(ds==@q4 ^ (p1 & ~p2 & ~p3)) then @q5
else if(ds==@q2 ^ (true)) then @q2
else if(ds==@q5 ^ (~p1 & ~p2 & ~p3)) then @q1
else if(ds==@q5 ^ (p3 & ~p1 & ~p2)) then @q3
else if(ds==@q5 ^ (p2 & ~p1 & ~p3)) then @q4
else if(ds==@q5 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2
else if(ds==@q5 ^ (p1 & ~p2 & ~p3 & ~p4)) then @q5
else if(ds==@q5 ^ (p1 & p4 & ~p2 & ~p3)) then @q6
else if(ds==@q6 ^ ((~p1 & ~p2) | (~p1 & ~p3) | (~p2 & ~p3))) then @q6
else if(ds==@q6 ^ ((p1 & p2) | (p1 & p3) | (p2 & p3))) then @q2'''

class MiniGridEnvOX1(MiniGridEnv):

    def __init__(self,
            reset_ps,
            size=15,
            agent_start_pos = (2, 4),
            agent_start_dir = 2,
            max_steps: int = 200,
            option_index: int = 0,
            **kwargs):
        self.agent_start_pos = (2, 4)  # agent的起始位置
        self.agent_start_dir = 2  # agent的起始朝向
        self.Ball1_pos = (5, 10)
        self.Ball2_pos = (13, 1)
        self.Ball3_pos = (13, 8)
        self.step_count = 0
        self.task = None
        self.Ball = None
        self.size = size
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.dfa = DFATransformer(dfa_text)
        self.option_index = option_index
        self.start = False
        self.reset_ps = reset_ps
        self.end_states = None
        self.buffer_action = None
        self.initial_state = None
        self.value = None
        self.log_prob = None


        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )


    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):

        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding and middle walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height//2)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width//2, 0)
        self.grid.vert_wall(width - 1, 0)
        # self.grid.set(width//2, 6, None)

        # Place the Door
        self.grid.set(width//2, 6, Door("yellow",is_open=True))
        self.grid.set(1, height//2, None)
        self.grid.set(1, height // 2, Door("yellow",is_open=True))
        self.grid.set(12, height // 2, None)
        self.grid.set(12, height // 2, Door("yellow",is_open=True))
        self.put_obj(Goal(),12,10)
        # Place the Ball
        self.grid.set(5, 10, Ball("blue"))
        self.grid.set(13, 1, Ball("blue"))
        self.grid.set(13, 8, Ball("blue"))
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)

        dfa_state = self.dfa.dfa_state.split('q')[1]
        dfa_index = int(dfa_state) - 1
        obs = {"image": image, "direction": self.agent_dir, "pos": self.agent_pos, 'ds': dfa_index}

        return obs

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        props = {'p1': False, 'p2': False, 'p3': False, 'p4': False}
        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()


        # a1
        if (8 <= self.agent_pos[0] <= 13 and 1 <= self.agent_pos[1] <= 6):
            props['p1'] = True
        elif (1 <= self.agent_pos[0] <= 6 and 8 <= self.agent_pos[1] <= 13):
            props['p2'] = True
        elif (8 <= self.agent_pos[0] <= 13 and 8 <= self.agent_pos[1] <= 13):
            props['p3'] = True

        if self.carrying is not None:
            props['p4'] = True

        # DFA状态机走一步
        last_dfa_state = self.dfa.dfa_state
        terminated1, if_success, if_failure = self.dfa.step(props)

        terminated = terminated or terminated1
        # reward shaping
        '''if not terminated and last_dfa_state != self.dfa.dfa_state:
            reward = 10
        elif not terminated and last_dfa_state == self.dfa.dfa_state:
            reward = 0
        elif if_failure:
            reward = -10
        elif if_success:
            reward = 100
        else:
            sys.exit('error!')'''
        if if_success:
            reward = 100
        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        assert self.reset_ps is not None
        reset_state = ray.get(self.reset_ps.get_state.remote(self.option_index))
        while not self.start:
            if reset_state is None:
                time.sleep(5)
                reset_state = ray.get(self.reset_ps.get_state.remote(self.option_index))
            else:
                self.start = True
                print(self.option_index, "start!!!")
                break

        info = reset_state
        obs, self.end_states, self.buffer_action, self.value, self.log_prob, env = info
        env = copy.deepcopy(env)
        # Reinitialize episode-specific variables
        self.agent_pos = env.get_wrapper_attr('agent_pos')
        self.agent_dir = env.get_wrapper_attr('agent_dir')
        self.put_obj(Goal(), 12, 10)
        self.grid = env.get_wrapper_attr('grid')
        self.step_count = env.get_wrapper_attr('step_count')
        self.carrying = env.get_wrapper_attr('carrying')
        self.dfa.dfa_state = '@q' + str(obs['ds'].item() + 1)

        # Return first observation
        obs = self.gen_obs()
        self.initial_state = obs
        # print(obs)
        return obs, {}




