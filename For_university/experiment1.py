import random

SIZE_OF_UNIVERSE = 100
NUMBER_OF_SUBSETS = 40
CARDINALITY = 4
REMAINING_SUBSET = NUMBER_OF_SUBSETS - CARDINALITY
ELEMENTS_OF_UNIVERSE = list(range(SIZE_OF_UNIVERSE))
RATIO = int(NUMBER_OF_SUBSETS / CARDINALITY)
NUMBER_OF_DUPLICATE = int((NUMBER_OF_SUBSETS-CARDINALITY)/(CARDINALITY/2))
UNCOVERED_ELEMENTS = set(ELEMENTS_OF_UNIVERSE)
TEST_LOOP = 1


def making_universe(SIZE_OF_UNIVERSE, CARDINALITY):
    universe = []
    i = 0
    small_subset = int(SIZE_OF_UNIVERSE / CARDINALITY - 1)
    big_subset = int(SIZE_OF_UNIVERSE / CARDINALITY + 1)
    last_element = 0
    for index in range(CARDINALITY):
        if index < int(CARDINALITY / 2):
            subset = list(ELEMENTS_OF_UNIVERSE[last_element:last_element + big_subset])
            for i in range(NUMBER_OF_DUPLICATE+1):
                universe.append(subset)
            last_element = last_element + big_subset
        else:
            universe.append(list(ELEMENTS_OF_UNIVERSE[last_element:last_element + small_subset]))
            last_element = last_element + small_subset


    return universe


def coverage_function(subset):
    return len(set(UNCOVERED_ELEMENTS).difference(set(subset)))


def find_best_in_segment(subsets_in_segment):
    best_subset = (min(subsets_in_segment, key=coverage_function))
    return best_subset


def proposed_algorithem(subsets):
    global UNCOVERED_ELEMENTS
    solution = list()
    for i in range(CARDINALITY):
        best_marginal_subset = find_best_in_segment(subsets[i * RATIO:i * RATIO + RATIO])
        solution.append(best_marginal_subset)
        UNCOVERED_ELEMENTS = UNCOVERED_ELEMENTS.difference(set(best_marginal_subset))
    return solution


def main():
    global UNCOVERED_ELEMENTS
    subsets = making_universe(SIZE_OF_UNIVERSE, CARDINALITY)
    print(subsets)
    results_sum = float(0)
    for i in range(TEST_LOOP):
        subsets = random.sample(subsets, NUMBER_OF_SUBSETS)
        solution = proposed_algorithem(subsets)
        results_sum += (1 - float(len(UNCOVERED_ELEMENTS) / SIZE_OF_UNIVERSE))
        UNCOVERED_ELEMENTS = set(ELEMENTS_OF_UNIVERSE)

    print('The expectation of the competetive ratio is %f' % float(results_sum/TEST_LOOP))






if __name__ == '__main__':
    main()
