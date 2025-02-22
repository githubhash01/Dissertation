from enum import Enum

class GradientMethod(Enum):
    AUTO_DIFF = 1
    FINITE_DIFFERENCE = 2
    IMPLICIT_DIFFERENTIATION = 3


# Set the default gradient method
GRADIENT_METHOD = GradientMethod.IMPLICIT_DIFFERENTIATION