from collections import defaultdict
from itertools import accumulate
import random

TEST_LOOP = 100000
ACCURACY = 0.01

sample_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
sample_weights = [0.1, 0.15, 0.4, 0.1, 0.05, 0.066666667, 0.133333333]


# TODO: Hess should implement!
def weighted_sampler(data, weights):
    """Should return a random item from data array, weighted by the weights array"""
    random_number = random.random()
    print('this is the: %f' % random_number)
    accumulated_list = accumulated_list_function(weights)
    print('This is the accumulated list: %s' % str(accumulated_list))
    for item_idx, curr_weight in enumerate(accumulated_list):
        if (random_number >= curr_weight) and (random_number <= accumulated_list[item_idx + 1]):
            print('this the data chosen: %s' % data[item_idx])
            return data[item_idx]


def accumulated_list_function(weights):
    return [0] + list((accumulate(weights)))


def close_enough(float1, float2):
    return abs(float1 - float2) <= ACCURACY


def main():
    results_count = defaultdict(int)
    for i in range(TEST_LOOP):
        item = weighted_sampler(sample_data, sample_weights)
        results_count[item] += 1
    for item, weight in zip(sample_data, sample_weights):
        assert close_enough(results_count[item] / TEST_LOOP, weight), "not accurate enough!"
    print('Great Success!!')


if __name__ == '__main__':
    main()
