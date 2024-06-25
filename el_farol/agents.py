import mesa

class BarCustomer(mesa.Agent):
    def __init__(self, unique_id, model, memory_size, crowd_threshold):
        super().__init__(unique_id, model)
        self.attend = False
        self.memory_size = memory_size
        self.crowd_threshold = crowd_threshold
        self.utility = 0

    def step(self):
        # Check if the customer's action is to attend the bar
        if self.model.action_dict[self.unique_id] == 1:
            self.attend = True
            self.model.attendance += 1
        else:
            self.attend = False
