# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: reserve global parameters
#
# Created at 9/16/22


UCB_BETA = 0.2


class VarTypes:
    INTEGER = 1
    BOOL = 2
    ENUM = 3
    FLOAT = 4
    ALL_TYPES = {INTEGER, BOOL, ENUM, FLOAT}

    @staticmethod
    def str2type(s):
        type_dict = {
            "INTEGER": 1,
            "BOOL": 2,
            "ENUM": 3,
            "FLOAT": 4
        }
        try:
            return type_dict[s]
        except:
            raise Exception(f"VarType does not support {s}")