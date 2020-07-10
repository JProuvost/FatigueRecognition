import math


def truncate(number, digits = 3) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper
