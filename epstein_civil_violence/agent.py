from . utility import move, update_neighbors
from mesa_models.epstein_civil_violence.agent import Citizen, Cop

class Citizen_RL(Citizen):
    def __init__(self, unique_id, model, pos, hardship, regime_legitimacy, risk_aversion, vision):
        super().__init__(unique_id, model, pos, hardship, regime_legitimacy, risk_aversion, 0, vision)
        self.prev_pos = pos
        self.prev_condition = "Quiescent"

    def step(self):
        self.prev_condition = self.condition
        # If in jail decrease sentence, else update condition
        if self.jail_sentence:
            self.jail_sentence -= 1
        else:
            # Update condition based on action
            self.condition = "Active" if self.model.action_dict[self.unique_id][0] == 1 else "Quiescent"
        self.prev_pos = self.pos
        if self.model.movement:
            self.neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, radius=self.vision)
            self.empty_neighbors = [c for c in self.neighborhood if self.model.grid.is_cell_empty(c)]
            move(self, self.model.action_dict[self.unique_id][1], self.empty_neighbors)
        # Update observation space
        update_neighbors(self)
        self.model.observation[self.unique_id] = self.neighbors


class Cop_RL(Cop):
    def __init__(self, unique_id, model, pos, vision):
        super().__init__(unique_id, model, pos, vision)
        self.neighbors_agent = []
        self.arrest_made = False

    def step(self):
        # Arrest if active citizen is indicated in action
        # Arrest is made based on previous states on which the action was decided
        self.neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, radius=self.vision)
        action_tuple = self.model.action_dict[self.unique_id]
        arrest_pos = self.neighborhood[action_tuple[0]]
        for agent in self.neighbors_agent:
            if agent.breed == "citizen" and agent.prev_condition == "Active" and agent.jail_sentence == 0 and agent.prev_pos == arrest_pos:
                sentence = self.random.randint(0, self.model.max_jail_term)
                agent.jail_sentence = sentence
                agent.condition = "Quiescent"
                self.arrest_made = True

        self.prev_pos = self.pos
        if self.model.movement:
            self.empty_neighbors = [c for c in self.neighborhood if self.model.grid.is_cell_empty(c)]
            move(self, self.model.action_dict[self.unique_id][1], self.empty_neighbors)
        # Update observation space
        update_neighbors(self)
        self.model.observation[self.unique_id] = self.neighbors
