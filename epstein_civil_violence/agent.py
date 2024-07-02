import math

import mesa


def move(self, action, empty_neighbors):
    # Get possible steps for the agent
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
    new_position = (new_position[0] % self.model.grid.width, new_position[1] % self.model.grid.height)
    # Check if the new position is valid, then move the agent
    if new_position in empty_neighbors:
        self.model.grid.move_agent(self, new_position)
    
def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors_agent = self.model.grid.get_cell_list_contents(self.neighborhood)

        # Get self.neighbors cells from self.model.obs_grid
        self.neighbors = [self.model.obs_grid[neighbor[0]][neighbor[1]] for neighbor in self.neighborhood]
        self.empty_neighbors = [c for c in self.neighborhood if self.model.grid.is_cell_empty(c)]

        
class Citizen(mesa.Agent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        hardship,
        regime_legitimacy,
        risk_aversion,
        vision,
    ):
        """
        Create a new Citizen.
        Args:
            unique_id: unique int
            x, y: Grid coordinates
            hardship: Agent's 'perceived hardship (i.e., physical or economic
                privation).' Exogenous, drawn from U(0,1).
            regime_legitimacy: Agent's perception of regime legitimacy, equal
                across agents.  Exogenous.
            risk_aversion: Exogenous, drawn from U(0,1).
            threshold: if (grievance - (risk_aversion * arrest_probability)) >
                threshold, go/remain Active
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(unique_id, model)
        self.breed = "citizen"
        self.pos = pos
        self.hardship = hardship
        self.regime_legitimacy = regime_legitimacy
        self.risk_aversion = risk_aversion
        self.condition = "Quiescent"
        self.vision = vision
        self.jail_sentence = 0
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_probability = None

    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """
        if self.jail_sentence:
            self.jail_sentence -= 1
        else:
            if self.model.action_dict[self.unique_id][0] == 1:
                self.condition = "Active"
            else:
                self.condition = "Quiescent"
        update_neighbors(self)
        if self.model.movement:
            move(self, self.model.action_dict[self.unique_id][1], self.empty_neighbors)

        self.model.observation[self.unique_id] =  self.neighbors

class Cop(mesa.Agent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
    """

    def __init__(self, unique_id, model, pos, vision):
        """
        Create a new Cop.
        Args:
            unique_id: unique int
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(unique_id, model)
        self.breed = "cop"
        self.pos = pos
        self.vision = vision
        self.neighbors_agent = []

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        action_tuple = self.model.action_dict[self.unique_id]
        arrest_pos = (self.pos[0] + action_tuple[0] // 3, self.pos[1] + action_tuple[0] % 3)
        for agent in self.neighbors_agent:
            if (
                agent.breed == "citizen"
                and agent.condition == "Active"
                and agent.jail_sentence == 0
                and agent.pos == arrest_pos
            ):
                sentence = self.random.randint(0, self.model.max_jail_term)
                agent.jail_sentence = sentence
                agent.condition = "Quiescent"
                
        update_neighbors(self)
        if self.model.movement:
            move(self, self.model.action_dict[self.unique_id][1], self.empty_neighbors)

        self.model.observation[self.unique_id] =  self.neighbors
