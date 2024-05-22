import numpy as np
from pymoo.core.duplicate import ElementwiseDuplicateElimination

import constants


class Duplicates(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        ind1 = [x.dish_types._get_column_array(constants.FOOD_ID) for x in a.X[0].days]
        ind2 = [x.dish_types._get_column_array(constants.FOOD_ID) for x in b.X[0].days]

        a_all = np.concatenate(ind1)
        b_all = np.concatenate(ind2)

        return np.equal(a_all, b_all).all()
