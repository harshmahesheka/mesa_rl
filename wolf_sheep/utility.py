def create_intial_agents(self, Sheep_RL, Wolf_RL, GrassPatch):
    # Create sheep:
    for i in range(self.initial_sheep):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        energy = self.random.randrange(2 * self.sheep_gain_from_food)
        unique_id_str = f"sheep_{self.next_id()}"
        sheep = Sheep_RL(unique_id_str, None, self, True, energy)
        self.grid.place_agent(sheep, (x, y))
        self.schedule.add(sheep)

    # Create wolves
    for i in range(self.initial_wolves):
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        energy = self.random.randrange(2 * self.wolf_gain_from_food)
        unique_id_str = f"wolf_{self.next_id()}"
        wolf = Wolf_RL(unique_id_str, None, self, True, energy)
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

            unique_id_str = f"grass_{self.next_id()}"
            patch = GrassPatch(unique_id_str, None, self, fully_grown, countdown)
            self.grid.place_agent(patch, (x, y))
            self.schedule.add(patch)

def move(self, action):
    # Get possible steps for the agent
    possible_steps = self.model.grid.get_neighborhood(
        self.pos,
        moore=True,
        include_center=False
    )
    action = int(action)  # Convert action to integer

    # Uncomment this to test random baselines
    # action = np.random.randint(0, 4)

    # Move the agent based on the action
    if action == 0:
        new_position = (self.pos[0] + 1, self.pos[1])
    elif action == 1:
        new_position = (self.pos[0] - 1, self.pos[1])
    elif action == 2:
        new_position = (self.pos[0], self.pos[1] - 1)
    elif action == 3:
        new_position = (self.pos[0], self.pos[1] + 1)

    # Check if the new position is valid, then move the agent
    if new_position in possible_steps:
        self.model.grid.move_agent(self, new_position)

def grid_to_observation(self, Sheep_RL, Wolf_RL, GrassPatch):
    # Convert grid to matrix for better representation
    self.obs_grid = []
    for i in self.grid._grid:
        row = []
        for j in i:
            value = [0, 0, 0]
            for agent in j:
                if isinstance(agent, Sheep_RL):
                    value[0] = 1
                elif isinstance(agent, Wolf_RL):
                    value[1] = 1
                elif isinstance(agent, GrassPatch) and agent.fully_grown:
                    value[2] = 1
            row.append(value)
        self.obs_grid.append(row)