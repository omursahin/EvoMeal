from abc import abstractmethod, ABC


class AbstractSelection(ABC):
    @abstractmethod
    def select(self, population):
        pass

    @staticmethod
    def get_better(ind_1, ind_2):
        return ind_1 if ind_1.total_fitness > ind_2.total_fitness else ind_2
