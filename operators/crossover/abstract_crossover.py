from abc import abstractmethod, ABC


class AbstractCrossover(ABC):

    @abstractmethod
    def crossover(self, individual_1, individual_2):
        pass
