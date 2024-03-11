import numpy as np
from agents import GrassPatch, Sheep, Wolf

def create_initial_agents(self):
    """
    Create initial agents including sheep, wolves, and grass patches.
    """
    # Create sheep:
    for i in range(self.initial_sheep):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        energy = self.random.randrange(2 * self.sheep_gain_from_food)
        sheep = Sheep(self.next_id(), (x, y), self, energy)
        self.grid.place_agent(sheep, (x, y))
        self.schedule.add(sheep)

    # Create wolves
    for i in range(self.initial_wolves):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        energy = self.random.randrange(2 * self.wolf_gain_from_food)
        wolf = Wolf(self.next_id(), (x, y), self, energy)
        self.grid.place_agent(wolf, (x, y))
        self.schedule.add(wolf)

    # Create grass patches
    if self.grass:
        for agent, (x, y) in self.grid.coord_iter():
            fully_grown = self.random.choice([True, False])

            if fully_grown:
                countdown = self.grass_regrowth_time
            else:
                countdown = self.random.randrange(self.grass_regrowth_time)

            patch = GrassPatch(self.next_id(), (x, y), self, fully_grown, countdown)
            self.grid.place_agent(patch, (x, y))
            self.schedule.add(patch)


def get_observation(self, current_agent):
    """
    Get the observation, which is a dictionary.
    The 'grid' key contains a 3D array with information about the agents at each grid location.
    The 'current_agent' key contains a dictionary with the current agent's position and energy.
    """
    grid = np.zeros((self.width, self.height, 4), dtype=bool)  # 3 for sheep, wolf, grass

    for x in range(self.width):
        for y in range(self.height):
            cell = self.grid.get_cell_list_contents([(x, y)])

            for agent in cell:
                if isinstance(agent, Sheep):
                    grid[x, y, 0] = True
                elif isinstance(agent, Wolf):
                    grid[x, y, 1] = True
                elif isinstance(agent, GrassPatch):
                    grid[x, y, 2] = agent.fully_grown

    sub_grid = get_subgrid(self, grid, current_agent, False)

    observation = {
        'grid': sub_grid,
        'energy': np.array([np.float32(current_agent.energy)])
    }

    return observation


def remove_dead_agents(self):
    """
    Remove dead agents from the grid and the schedule.
    """
    for agent in list(self.schedule.agents):
        if isinstance(agent, (Sheep, Wolf)) and not agent.living:
            self.grid.remove_agent(agent)
            self.schedule.remove(agent)
    self.agents = [a.unique_id for a in self.schedule.agents if isinstance(a, (Sheep, Wolf))]


def get_subgrid(self, grid, agent, default_value):
    """
    Get a sub-grid around the agent's position.
    """
    # Identify the agent's position
    agent_position = agent.pos

    # Determine the size of the sub-grid
    sub_grid_size = (10, 10)

    # Calculate the top-left and bottom-right coordinates of the sub-grid
    top_left = (int(agent_position[0] - sub_grid_size[0] / 2), int(agent_position[1] - sub_grid_size[1] / 2))
    bottom_right = (top_left[0] + sub_grid_size[0], top_left[1] + sub_grid_size[1])

    # If the agent is near the edge of the grid, adjust the coordinates and note how much padding is needed
    padding = [0, 0, 0, 0]  # top, bottom, left, right
    if top_left[0] < 0:
        padding[0] = -top_left[0]
        top_left = (0, top_left[1])
    if top_left[1] < 0:
        padding[2] = -top_left[1]
        top_left = (top_left[0], 0)
    if bottom_right[0] > self.grid.width:
        padding[1] = bottom_right[0] - self.grid.width
        bottom_right = (self.grid.width, bottom_right[1])
    if bottom_right[1] > self.grid.height:
        padding[3] = bottom_right[1] - self.grid.height
        bottom_right = (bottom_right[0], self.grid.height)

    # Extract the sub-grid
    sub_grid = grid[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # If padding is needed, pad the sub-grid with the default value
    if any(padding):
        sub_grid = np.pad(sub_grid, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0)), 'constant', constant_values=default_value)

    for i in range(len(sub_grid)):
        for j in range(len(sub_grid[i])):
            if all(sub_grid[i][j] == [True, True, True, True]):
                sub_grid[i][j] = [False, False, False, True]

    return sub_grid