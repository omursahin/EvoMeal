import numpy as np

from helper.config import Config


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class Random:
    def __init__(self):
        conf = Config()

        self.random = np.random
        if not conf.RANDOM_SEED:
            self.random.seed(conf.SEED)

    def random_float(self):
        return self.random.rand()

    def random_bool(self, value):
        return self.random.rand() < value

    def random_int(self, max):
        return self.random.randint(0, max)

    def random_choice(self, max_size, selection_probs):
        return self.random.choice(max_size, p=selection_probs)
