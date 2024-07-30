def create_intial_agents(self, Citizen_RL, Cop_RL):
    # Create agents
    unique_id = 0
    if self.cop_density + self.citizen_density > 1:
        raise ValueError("Cop_RL density + citizen density must be less than 1")
    for contents, (x, y) in self.grid.coord_iter():
        if self.random.random() < self.cop_density:
            unique_id_str = f"cop_{unique_id}"
            cop = Cop_RL(unique_id_str, self, (x, y), vision=self.cop_vision)
            unique_id += 1
            self.grid[x][y] = cop
            self.schedule.add(cop)
        elif self.random.random() < (self.cop_density + self.citizen_density):
            unique_id_str = f"citizen_{unique_id}"
            citizen = Citizen_RL(
                unique_id_str,
                self,
                (x, y),
                hardship=self.random.random(),
                regime_legitimacy=self.legitimacy,
                risk_aversion=self.random.random(),
                vision=self.citizen_vision,
            )
            unique_id += 1
            self.grid[x][y] = citizen
            self.schedule.add(citizen)

def grid_to_matrix(self, Citizen_RL):
    # Convert neighborhood to observation grid
    self.obs_grid = []
    for i in self.grid._grid:
        row = []
        for j in i:
            if j is None:
                row.append(0)  # Empty cell
            elif isinstance(j, Citizen_RL):
                if j.condition == "Quiescent":
                    row.append(3 if j.jail_sentence > 0 else 1)  # Quiescent citizen (jailed or not)
                elif j.condition == "Active":
                    row.append(2)  # Active citizen
            else:
                row.append(4)  # Cop
        self.obs_grid.append(row)

def move(self, action, empty_neighbors):
    action = int(action)
    if action == 0:
        new_position = (self.pos[0] + 1, self.pos[1])  # Move right
    elif action == 1:
        new_position = (self.pos[0] - 1, self.pos[1])  # Move left
    elif action == 2:
        new_position = (self.pos[0], self.pos[1] - 1)  # Move up
    elif action == 3:
        new_position = (self.pos[0], self.pos[1] + 1)  # Move down
    new_position = (new_position[0] % self.model.grid.width, new_position[1] % self.model.grid.height)  # Wrap around the grid
    if new_position in empty_neighbors:
        self.model.grid.move_agent(self, new_position)  # Move to the new position

def update_neighbors(self):
    self.neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, radius=self.vision)  # Get the Moore neighborhood
    self.neighbors_agent = self.model.grid.get_cell_list_contents(self.neighborhood)  # Get the contents of the neighborhood cells
    self.neighbors = [self.model.obs_grid[neighbor[0]][neighbor[1]] for neighbor in self.neighborhood]  # Get the values from the observation grid for the neighborhood cells
    self.empty_neighbors = [c for c in self.neighborhood if self.model.grid.is_cell_empty(c)]  # Get the empty cells in the neighborhood
