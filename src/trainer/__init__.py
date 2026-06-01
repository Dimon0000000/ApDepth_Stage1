# Author: Bingxin Ke
# Last modified: 2024-05-17

from .trainer_s1 import ApDepthTrainerS1


trainer_cls_name_dict = {
    "ApDepthTrainerS1": ApDepthTrainerS1,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
