import mesa
import numpy as np

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

class Sheep(mesa.Agent):
    """
    A sheep that walks around, reproduces (asexually) and gets eaten.
    """

    energy = None

    def __init__(self, unique_id, pos, model, energy=None):
        super().__init__(unique_id, model)
        self.energy = energy
        self.done = False
        self.pos = pos
        self.living = True
        self.time = 0

    def step(self, action):
        """
        A model step. Move, then eat grass and reproduce.
        """
        if self.living:
            self.time += 1
            move(self, action)

            if self.model.grass:
                # Reduce energy
                self.energy -= 1

                # If there is grass available, eat it
                this_cell = self.model.grid.get_cell_list_contents([self.pos])
                grass_patch = next(obj for obj in this_cell if isinstance(obj, GrassPatch))
                if grass_patch.fully_grown:
                    self.energy += self.model.sheep_gain_from_food
                    grass_patch.fully_grown = False

                # Death
                if self.energy < 0:
                    self.living = False

            if self.living and self.random.random() < self.model.sheep_reproduce:
                # Create a new sheep:
                if self.model.grass:
                    self.energy /= 2
                lamb = Sheep(
                    self.model.next_id(), self.pos, self.model, self.energy
                )
                self.model.grid.place_agent(lamb, self.pos)
                self.model.schedule.add(lamb)


class Wolf(mesa.Agent):
    """
    A wolf that walks around, reproduces (asexually) and eats sheep.
    """

    energy = None

    def __init__(self, unique_id, pos, model, energy=None):
        super().__init__(unique_id, model)
        self.energy = energy
        self.done = False
        self.pos = pos
        self.living = True
        self.time = 0

    def step(self, action):
        if self.living:
            self.time += 1
            move(self, action)
            self.energy -= 1

            # If there are sheep present, eat one
            x, y = self.pos
            this_cell = self.model.grid.get_cell_list_contents([self.pos])
            sheep = [obj for obj in this_cell if isinstance(obj, Sheep) and obj.living]
            if len(sheep) > 0:
                sheep_to_eat = self.random.choice(sheep)
                self.energy += self.model.wolf_gain_from_food

                sheep_to_eat.living = False

            # Death or reproduction
            if self.energy < 0:
                self.living = False
            else:
                if self.random.random() < self.model.wolf_reproduce:
                    # Create a new wolf cub
                    self.energy /= 2
                    cub = Wolf(
                        self.model.next_id(), self.pos, self.model, self.energy
                    )
                    self.model.grid.place_agent(cub, cub.pos)
                    self.model.schedule.add(cub)


class GrassPatch(mesa.Agent):
    """
    A patch of grass that grows at a fixed rate and it is eaten by sheep
    """

    def __init__(self, unique_id, pos, model, fully_grown, countdown):
        """
        Creates a new patch of grass

        Args:
            grown: (boolean) Whether the patch of grass is fully grown or not
            countdown: Time for the patch of grass to be fully grown again
        """
        super().__init__(unique_id, model)
        self.fully_grown = fully_grown
        self.countdown = countdown
        self.pos = pos

    def step(self):
        if not self.fully_grown:
            if self.countdown <= 0:
                # Set as fully grown
                self.fully_grown = True
                self.countdown = self.model.grass_regrowth_time
            else:
                self.countdown -= 1
