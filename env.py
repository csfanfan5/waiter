import numpy as np

class Restaurant:
    def __init__(self, w, h, tables, v, p, wall_penalty=0.1, table_reward=1, time_penalty=0.00001, time_penalty_type="linear"):
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
        self.v = v
        self.p = p
        self.wall_penalty = wall_penalty
        self.table_reward = table_reward
        self.time_penalty = time_penalty
        self.time_penalty_type = time_penalty_type

        # randomly place agent in restaurant to begin
        self.agent = [np.random.uniform(0, self.w), np.random.uniform(0, self.h)] 

    def reset(self):
        self.times = len(self.tables) * [-1]
        self.agent = [np.random.uniform(0, self.w), np.random.uniform(0, self.h)] 
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
            self.agent[0] = min(max(self.agent[0], 0), self.w)
            self.agent[1] = min(max(self.agent[1], 0), self.h)
            reward -= self.wall_penalty
        
        for i in range(len(self.tables)):
            # if table is currently empty, regenerate with probability p
            if self.times[i] == -1 and not (self.tables[i][0] <= self.agent[0] <= self.tables[i][1] and self.tables[i][2] <= self.agent[1] <= self.tables[i][3]):
                if np.random.binomial(1, self.p):
                    self.times[i] = 0
            # if table is filled
            else:
                self.times[i] += 1 # time increment
                
                if self.time_penalty_type == "const":    
                    reward -= self.time_penalty
                elif self.time_penalty_type == "linear":  
                    reward -= self.time_penalty * self.times[i]
                elif self.time_penalty_type == "exp":  
                    reward -= self.time_penalty * np.exp(self.times[i]) 
                else:
                    raise Exception("Not valid type.")

                if self.tables[i][0] <= self.agent[0] <= self.tables[i][1] and self.tables[i][2] <= self.agent[1] <= self.tables[i][3]:
                    reward += self.table_reward
                    self.times[i] = -1 
        
        return self.times, self.agent, reward


        

        
        