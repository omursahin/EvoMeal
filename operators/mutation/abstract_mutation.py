from abc import abstractmethod, ABC


class AbstractMutation(ABC):

    @abstractmethod
    def mutate(self, individual, get_one_dish_type):
        pass
