from mesa_models.wolf_sheep.agents import Sheep, Wolf, GrassPatch
from . utility import move

class Sheep_RL(Sheep):

    def random_move(self):
        """
        We change random walk to take action given by model
        """
        action = self.model.action_dict[self.unique_id]
        move(self, action)


class Wolf_RL(Wolf):

    def random_move(self):
        """
        We change random walk to take action given by model
        """
        action = self.model.action_dict[self.unique_id]
        move(self, action)