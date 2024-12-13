import numpy as np
import gym
from gym import spaces
import numpy as np
import math

# gym wrapper for the restaurant environment
class RestaurantEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, w, h, tables, v, p, wall_penalty=0, table_reward=0, time_reward=0.1,  
                 table_threshold=10, table_waiting_time=300):
        super(RestaurantEnv, self).__init__()
        self.restaurant = Restaurant(
            w, h, tables, v, p, wall_penalty=wall_penalty, table_reward=table_reward, 
            time_reward=time_reward, table_threshold=table_threshold, table_waiting_time=table_waiting_time
        )
        
        self.n_tables = len(tables)
        self.obs_dim = self.n_tables + 2  # times for each table + agent x,y
        
        # Action space: one continuous angle in [0, 2*pi]
        self.action_space = spaces.Box(low=0.0, high=2.0*math.pi, shape=(1,), dtype=np.float32)
        
        # Observation space: times + agent position.
        # times range from -1 to table_waiting_time, agent coords in [0,w]x[0,h].
        # For simplicity, we’ll just use large bounds or inf. 
        # If we want tighter bounds:
        low = np.array([ -1 ] * self.n_tables + [0, 0], dtype=np.float32)
        high = np.array([ table_waiting_time ] * self.n_tables + [self.restaurant.w, self.restaurant.h], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        times, agent = self.restaurant.reset()
        obs = np.array(times + agent, dtype=np.float32)
        return obs

    def step(self, action):
        # action is an array of size 1, containing the angle
        alpha = float(action[0])
        times, agent, reward, done = self.restaurant.step(alpha)
        obs = np.array(times + agent, dtype=np.float32)
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        # Implement visualization if desired
        pass

    def close(self):
        pass

# Example of usage:
# env = RestaurantEnv(w=100, h=100, tables=[[10,20,30,40],[50,60,70,80]], v=1.0, p=0.01)
# obs = env.reset()
# for _ in range(10):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     if done:
#         break
# env.close()

class Restaurant:
    def __init__(self, w, h, tables, v, p, wall_penalty=100, table_reward=100, time_reward=0.1,  
                 table_threshold=10, table_waiting_time=300):
        """
        Initialize the Restaurant

        Args:
            w: x - width of restaurant
            h: y - height of restaurant
            tables: list of lists [x1, x2, y1, y2] with x2 > x1, y2 > y1
                                holding borders of each table
            v: speed of waiter, assumed constant
            p: empty tables regenerate with some uniform probability to simulate expo process
            wall_penalty: penalty for hitting wall
            table_reward: reward for serving table
            time_penalty: prefactor of penalty on table timers
            time_penalty_type: in ["const", "linear", "exp"], default "linear"
        """
        self.w = w
        self.h = h
        self.tables = tables
        self.times = len(tables) * [-1]
        self.active = len(tables) * [False]
        self.v = v
        self.p = p
        self.wall_penalty = wall_penalty
        self.table_reward = table_reward
        self.time_reward=  time_reward
        self.table_threshold = table_threshold
        self.missed_tables = 0
        self.table_waiting_time = table_waiting_time

        # place the agent in the center of the restaurant 
        self.agent = [self.w / 2, self.h / 2]
        # self.agent = [np.random.uniform(0, self.w), np.random.uniform(0, self.h)] 

    def reset(self):
        self.times = len(self.tables) * [-1]
        # self.agent = [np.random.uniform(0, self.w), np.random.uniform(0, self.h)] 
        self.agent = [self.w / 2, self.h / 2]
        self.missed_tables = 0
        return (self.times, self.agent)

    def step(self, alpha):
        """
        Move foward by one timestep

        Args: (action)
            alpha: the angle at which the agent moves relative to positive x axis

        Returns:
            self.agent: agent coordinates [x,y]
            self.times: list of time waiting on each table
            reward: reward accumulated in this time step
        """
        reward = 0.0

        # move agent
        self.agent[0] += self.v * np.cos(alpha)
        self.agent[1] += self.v * np.sin(alpha)
        
        # if hits the wall, clamp the agent at the wall and incur penalty
        if not (0 <= self.agent[0] <= self.w and 0 <= self.agent[1] <= self.h):
            self.agent[0] = np.clip(self.agent[0], 0, self.w)
            self.agent[1] = np.clip(self.agent[1], 0, self.h)
            reward -= self.wall_penalty
        
        for i in range(len(self.tables)):
            # if the agent is on the table, serve it 
            if self.tables[i][0] <= self.agent[0] <= self.tables[i][1] and self.tables[i][2] <= self.agent[1] <= self.tables[i][3] and self.active[i]:
                reward += self.table_reward *  self.times[i] ** 2
                self.times[i] = -1 
                self.active[i] = False 


            # if table is currently empty, regenerate with probability p
            if self.active[i] == False:
                # if the waiter is late, mark it as missed 
                # respawn the table with probability p
                if np.random.binomial(1, self.p):
                    self.active[i] = True 
                    self.times[i] = 0

            # if the table has been waiting for longer than the threshold, 
            # mark it as missed 
            elif self.times[i] > self.table_waiting_time:
                # print("missed table")
                self.missed_tables += 1
                self.times[i] = -1
                self.active[i] = False

            # if the agent is active, update the timer on it 
            elif self.active[i]: 
                self.times[i] += 1 # time increment
                



        # check if we terminate 
        terminated = False
        if self.missed_tables >= self.table_threshold:
            terminated = True 
        else: 
            reward += self.time_reward 

        
        return self.times, self.agent, reward, terminated 
