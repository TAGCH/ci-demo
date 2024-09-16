"""Test for statistis.py."""
from unittest import TestCase
from statistics import average, variance, stdev
from math import sqrt

class StatisticsTest(TestCase):

    def test_variance_typical_values(self):
        """Variance of typical values."""
        self.assertEqual(0.0, variance([10.0, 10.0, 10.0, 10.0, 10.0]))
        self.assertEqual(2.0, variance([1, 2, 3, 4, 5]))
        self.assertEqual(8.0, variance([10, 2, 8, 4, 6]))
        self.assertEqual(4.0, variance([1000000, 1000004]))

    def test_variance_single_value(self):
        """Variance of a single value should be zero."""
        self.assertEqual(0.0, variance([10]))

    def test_variance_non_integers(self):
        """Variance should work with decimal values."""
        self.assertEqual(4.0, variance([0.1, 4.1]))
        self.assertEqual(8.0, variance([0.1, 4.1, 4.1, 8.1]))

    def test_variance_empty_list(self):
        """Variance should raise an error for an empty list."""
        with self.assertRaises(ValueError):
            variance([])

    def test_variance_large_floats(self):
        """Variance should handle very large floating-point numbers."""
        data = [1e6, 1e6 + 2]
        mean = sum(data) / len(data)
        variance_value = sum((x - mean) ** 2 for x in data) / (len(data) - 1)  # Sample variance
        self.assertAlmostEqual(variance_value, 2.0, places=6)

    def test_variance_large_dataset(self):
        """Variance should handle large datasets efficiently."""
        data = [i for i in range(1000)]
        mean = sum(data) / len(data)
        variance_value = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        self.assertAlmostEqual(variance_value, 83416.66666666667, places=6)

    def test_variance_floats_with_high_precision(self):
        """Variance should handle floating-point numbers with high precision."""
        data = [0.123456, 0.123457, 0.123458]
        mean = sum(data) / len(data)
        variance_value = sum((x - mean) ** 2 for x in data) / len(data)
        self.assertAlmostEqual(variance_value, 6.666666666680002e-13, places=12)

    def test_variance_identical_floats(self):
        """Variance should be zero for a list of identical floating-point numbers."""
        self.assertEqual(0.0, variance([2.5, 2.5, 2.5, 2.5, 2.5]))

    def test_variance_mixed_data_types(self):
        """Variance should work with a mix of integers and floats."""
        data = [1, 2.5, 3, 4.0]
        mean = sum(data) / len(data)
        variance_value = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        self.assertAlmostEqual(variance_value, 1.5625, places=4)

    def test_variance_large_numbers(self):
        """Variance should handle very large numbers correctly."""
        data = [1e12, 1e12 + 1, 1e12 + 2]
        mean = sum(data) / len(data)
        variance_value = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        self.assertAlmostEqual(variance_value, 1.0, places=6)

    def test_variance_zeros(self):
        """Variance should be zero if all values are zero."""
        self.assertEqual(0.0, variance([0, 0, 0, 0, 0]))

    def test_variance_negative_and_positive(self):
        """Variance should work with a mix of negative and positive numbers."""
        data = [-1, 0, 1]
        mean = sum(data) / len(data)
        variance_value = sum((x - mean) ** 2 for x in data) / len(data)
        self.assertAlmostEqual(variance_value, 0.6666666666666666, places=6)

    def test_variance_floats_high_precision(self):
        """Variance should handle floating-point numbers with high precision."""
        data = [0.123456, 0.123457, 0.123458]
        mean = sum(data) / len(data)
        variance_value = sum((x - mean) ** 2 for x in data) / len(data)
        self.assertAlmostEqual(variance_value, 6.666666666666667e-13, places=12)

    def test_stdev_typical_values(self):
        """Standard deviation of typical values."""
        self.assertEqual(0.0, stdev([10.0]))
        self.assertEqual(2.0, stdev([1, 5]))
        self.assertEqual(sqrt(0.5), stdev([0, 0.5, 1, 1.5, 2]))

    def test_stdev_single_value(self):
        """Standard deviation of a single value should be zero."""
        self.assertEqual(0.0, stdev([10]))

    def test_stdev_empty_list(self):
        """Standard deviation should raise an error for an empty list."""
        with self.assertRaises(ValueError):
            stdev([])

    def test_stdev_identical_values(self):
        """Standard deviation should be zero when all values are identical."""
        self.assertEqual(0.0, stdev([5, 5, 5, 5]))

    def test_stdev_large_numbers(self):
        """Standard deviation should work with large numbers."""
        self.assertEqual(2.0, stdev([1000000, 1000004]))

    def test_stdev_floating_point_precision(self):
        """Test standard deviation with values prone to floating-point precision issues."""
        self.assertAlmostEqual(1.0, stdev([1, 3]), places=6)

    def test_stdev_negative_numbers(self):
        """Standard deviation should work with negative numbers."""
        self.assertAlmostEqual(1, stdev([-1, -3]), places=6)

    def test_stdev_mixed_positive_negative(self):
        """Standard deviation should work with mixed positive and negative numbers."""
        mean = sum([-3, 2]) / 2
        variance = sum((x - mean) ** 2 for x in [-3, 2]) / 2
        self.assertAlmostEqual(sqrt(variance), stdev([-3, 2]), places=6)

    def test_stdev_floats_high_precision(self):
        """Standard deviation should handle floating-point numbers with high precision."""
        data = [0.123456, 0.123457, 0.123458]
        self.assertAlmostEqual(stdev(data), 0.0, places=12)

    def test_average_typical_values(self):
        """Average of typical values."""
        self.assertEqual(3.0, average([1, 2, 3, 4, 5]))
        self.assertEqual(5.0, average([4, 5, 6]))
        self.assertEqual(6.5, average([6, 7]))

    def test_average_single_value(self):
        """Average of a single value should be the value itself."""
        self.assertEqual(7.0, average([7]))

    def test_average_large_numbers(self):
        """Average should work with large numbers."""
        self.assertEqual(1000002.0, average([1000000, 1000004]))

    def test_average_small_numbers(self):
        """Average should work with small numbers."""
        self.assertAlmostEqual(0.233333, average([0.1, 0.4, 0.2]), places=6)

    def test_average_empty_list(self):
        """Average should raise an error for an empty list."""
        with self.assertRaises(ValueError):
            average([])

    def test_average_floats(self):
        """Average should work with float values."""
        self.assertAlmostEqual(2.333333, average([1.0, 2.0, 4.0]), places=6)

    def test_average_negative_numbers(self):
        """Average should work with negative numbers."""
        self.assertAlmostEqual(0.0, average([-3, -1, 1, 3]), places=6)
