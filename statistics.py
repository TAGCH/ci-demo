"""Basic statistical functions: average, variance, and standard deviation."""

from math import sqrt


def average(data):
    """Return the average of a list of numeric values in data.

    :param data: List of numeric values.
    :returns: The average of the values in data.
    :raises ValueError: If the list is empty.

    >>> average([])
    Traceback (most recent call last):
        ...
    ValueError: List must contain at least one value
    >>> average([1])
    1.0
    >>> average([1, 1, 1, 1])
    1.0
    >>> average([1, 2])
    1.5
    >>> average([1000000, 1000004])
    1000002.0
    """
    if len(data) == 0:
        raise ValueError("List must contain at least one value")
    return sum(data) / len(data)


def variance(data):
    """Compute the population variance of a list of numbers in data.

    The variance is the sum of squared differences between data values
    and their mean, divided by the number of items in the list.
    This differs from statistics.variance, which returns sample variance
    (where the sum is divided by n-1).

    Example: variance([1, 5]) is ((1 - 3)**2 + (5 - 3)**2) / 2 = 4.

    :param data: List of numbers for which variance will be computed.
           Must contain at least one element.
    :returns: Population variance of values in data list.
    :raises ValueError: If the data parameter is empty.

    >>> variance([])
    Traceback (most recent call last):
        ...
    ValueError: List must contain at least one value
    >>> variance([1])
    0.0
    >>> variance([1, 1, 1, 1])
    0.0
    >>> variance([1, 2])
    0.25
    >>> variance([1000000, 1000004])
    4.0
    """
    n = len(data)
    if n == 0:
        raise ValueError("List must contain at least one value")
    avg = average(data)
    # Calculate variance and round to avoid floating-point precision issues
    return round(sum((x - avg) ** 2 for x in data) / n, 6)


def stdev(data):
    """Compute the standard deviation of a list of values.

    :param data: List of numbers for which the standard deviation
                 will be computed. Must contain at least one element.
    :returns: Standard deviation of the values in the data list.
    :raises ValueError: If the data parameter is empty.

    >>> stdev([])
    Traceback (most recent call last):
        ...
    ValueError: List must contain at least one value
    >>> stdev([1])
    0.0
    >>> stdev([1, 1, 1, 1])
    0.0
    >>> stdev([1, 2])
    0.5
    >>> stdev([1000000, 1000004])
    2
    """
    if len(data) == 0:
        raise ValueError("List must contain at least one value")
    return sqrt(variance(data))
# type: ignore
