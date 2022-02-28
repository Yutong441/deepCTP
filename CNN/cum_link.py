# adapted from https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/models.py
from copy import deepcopy
import torch
from torch import nn

class LogisticCumulativeLink(nn.Module):
    """
    Converts a single number to the proportional odds of belonging to a class.
    Parameters
    ----------
    num_classes : int
        Number of ordered classes to partition the odds into.
    init_cutpoints : str (default='ordered')
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    """

    def __init__(self, num_classes: int, init_cutpoints: str 
                 = 'random', prob_fun='sigmoid') -> None:
        assert num_classes > 2, (
            'Only use this model if you have 3 or more classes'
        )
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        self.prob_fun = prob_fun
        if init_cutpoints == 'ordered':
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints[None])
        elif init_cutpoints == 'random':
            cutpoints = torch.rand(self.num_classes - 1)
            self.cutpoints = nn.Parameter(cutpoints[None])
        elif init_cutpoints == 'interval':
            cutpoints = torch.ones ([self.num_classes - 1]).float ()
            cutpoints [0] = - (self.num_classes - 1)/2
            self.cutpoints = nn.Parameter(cutpoints[None])
        else:
            raise ValueError(f'{init_cutpoints} is not a valid init_cutpoints '
                             f'type')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.prob_fun == 'sigmoid':
            sigmoids = torch.sigmoid(self.cutpoints - X)
            link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
            link_mat = torch.cat((
                    sigmoids[:, [0]], link_mat, (1 - sigmoids[:, [-1]])
                ), dim=1)
            return link_mat
        # https://home.ttic.edu/~nati/Publications/RennieSrebroIJCAI05.pdf
        elif self.prob_fun == 'gauss':
            return torch.exp (-((self.cutpoints - X)**2)/2)

class OrdinalLogisticModel(nn.Module):
    """
    "Wrapper" model for outputting proportional odds of ordinal classes.
    Pass in any model that outputs a single prediction value, and this module
    will then pass that model through the LogisticCumulativeLink module.
    Parameters
    ----------
    predictor : nn.Module
        When called, must return a torch.FloatTensor with shape [batch_size, 1]
    init_cutpoints : str (default='ordered')
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    """

    def __init__(self, predictor: nn.Module, cf) -> None:
        super().__init__()
        self.predictor = deepcopy(predictor)
        self.link = LogisticCumulativeLink(cf['predict_class'],
                        prob_fun=cf['prob_fun'])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.link(self.predictor(X))
