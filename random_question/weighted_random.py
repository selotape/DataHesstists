from collections import defaultdict
import random
TEST_LOOP = 1000000
ACCURACY = 0.001

sample_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
sample_weights = [0.1, 0.15, 0.4, 0.1, 0.05, 0.066666667, 0.133333333]



# TODO: Hess should implement!
def weighted_sampler(data, weights):
    random_number=random.random()
    print('this is the:')
    print(random_number)
    Dfunction = arrange_array(weights)
    print(Dfunction)
    for x in range(len(Dfunction)):
            if random_number >= Dfunction[x] and random_number <= Dfunction[x+1] :
                print('this the data chosen' )
                print(data[x])
                return data[x]




def arrange_array(weights):
    tmp=list(weights)
    for item in range(len(tmp)):
        if item != 0:
            tmp[item]=tmp[item] + tmp[item-1]
    return [0] + tmp

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

