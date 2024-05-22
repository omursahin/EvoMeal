from abc import abstractmethod, ABC


class AbstractFitness(ABC):

    @abstractmethod
    def fitness(self, individual):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_description(self):
        pass
