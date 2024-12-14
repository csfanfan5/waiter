import numpy as np

class Restaurant:
    def __init__(self, w, h, tables, v, p, wall_penalty=0.1, table_reward=25, time_penalty=0.00002, time_penalty_type="quadratic", 
                 unserved_threshold=500, large_unserved_penalty=10):
        """
        Initialize the Restaurant

        Args:
            w: x - width of restaurant
            h: y - height of restaurant
            tables: list of lists [x1, x2, y1, y2] specifying each table
            v: agent speed
            p: probability of table regeneration (not currently used in this code)
            wall_penalty: penalty for hitting wall
            table_reward: reward for serving a table
            time_penalty: prefactor of penalty on table timers
            time_penalty_type: one of ["const", "linear", "exp", "quadratic"]
            unserved_threshold: waiting time threshold after which unserved tables incur large penalty
            large_unserved_penalty: the large penalty added each step once threshold is exceeded
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
        self.unserved_threshold = unserved_threshold
        self.large_unserved_penalty = large_unserved_penalty

        # randomly place agent in the restaurant to begin
        self.agent = [np.random.uniform(0, self.w), np.random.uniform(0, self.h)]

    def reset(self):
        self.times = len(self.tables) * [1]
        self.agent = [np.random.uniform(0, self.w), np.random.uniform(0, self.h)]
        return (self.times, self.agent)

    def step(self, action):
        """
        Move forward by one timestep

        Args:
            action: integer index of the chosen table to move towards

        Returns:
            self.times: updated list of wait times for each table
            self.agent: agent coordinates [x,y]
            reward: reward accumulated in this time step
        """
        # compute direction to chosen table
        action = int(action)
        table = self.tables[action]
        center_x = (table[0] + table[1]) / 2.0
        center_y = (table[2] + table[3]) / 2.0
        dx = center_x - self.agent[0]
        dy = center_y - self.agent[1]
        alpha = np.arctan2(dy, dx)

        reward = 0.0
        # move agent in the direction of chosen table
        self.agent[0] += self.v * np.cos(alpha)
        self.agent[1] += self.v * np.sin(alpha)

        # if hits the wall, clamp the agent at the wall and incur penalty
        if not (0 <= self.agent[0] <= self.w and 0 <= self.agent[1] <= self.h):
            self.agent[0] = min(max(self.agent[0], 0), self.w)
            self.agent[1] = min(max(self.agent[1], 0), self.h)
            reward -= self.wall_penalty

        for i in range(len(self.tables)):
            # if table is filled, increment waiting time and apply penalties
            if self.times[i] != -1:
                self.times[i] += 1

                # apply base time penalty
                if self.time_penalty_type == "const":
                    reward -= self.time_penalty
                elif self.time_penalty_type == "linear":
                    reward -= self.time_penalty * self.times[i]
                elif self.time_penalty_type == "exp":
                    reward -= self.time_penalty * np.exp(self.times[i])
                elif self.time_penalty_type == 'quadratic':
                    reward -= self.time_penalty * (self.times[i] ** 2)
                else:
                    raise Exception("Not valid time_penalty_type.")

                # if table remains unserved beyond the threshold, apply large penalty
                if self.times[i] > self.unserved_threshold:
                    reward -= self.large_unserved_penalty

                # if agent is on the table, serve it and get reward
                if (self.tables[i][0] <= self.agent[0] <= self.tables[i][1]) and \
                   (self.tables[i][2] <= self.agent[1] <= self.tables[i][3]) and self.times[i] != -1:
                    # serve the table]
                    reward += self.table_reward
                    self.times[i] = -1
            else:
                # random p
                if np.random.binomial(1, self.p):
                    self.times[i] = 0

        return self.times, self.agent, reward
