import random

SIZE_OF_UNIVERSE = 4
NUMBER_OF_SUBSETS = 4
CARDINALITY = 2
REMAINING_SUBSET = NUMBER_OF_SUBSETS - CARDINALITY
ELEMENTS_OF_UNIVERSE = list(range(SIZE_OF_UNIVERSE))
RATIO = int(NUMBER_OF_SUBSETS / CARDINALITY)
# ratio has to be integer
NUMBER_OF_DUPLICATE = int((NUMBER_OF_SUBSETS - CARDINALITY) / (CARDINALITY / 2))
TEST_LOOP = 1


def make_universe():
    universe = []
    small_subset = int(SIZE_OF_UNIVERSE / CARDINALITY - 1)
    big_subset = int(SIZE_OF_UNIVERSE / CARDINALITY + 1)
    last_element = 0
    for index in range(CARDINALITY):
        if index < int(CARDINALITY / 2):
            subset = list(ELEMENTS_OF_UNIVERSE[last_element:last_element + big_subset])
            for i in range(NUMBER_OF_DUPLICATE + 1):
                universe.append(subset)
            last_element = last_element + big_subset
        else:
            universe.append(list(ELEMENTS_OF_UNIVERSE[last_element:last_element + small_subset]))
            last_element = last_element + small_subset

    return universe


def coverage_routine(subset, uncovered_elements):
    return len(uncovered_elements.difference(set(subset)))


def find_best_in_segment(subsets_in_segment, uncovered_elements):
    best_subset = (min(subsets_in_segment, key=lambda subset: coverage_routine(subset, uncovered_elements)))
    print('This is the best subset' + str(best_subset))
    return best_subset


def proposed_algorithm(subsets):
    uncovered_elements = set(ELEMENTS_OF_UNIVERSE)
    solution = []
    for i in range(CARDINALITY):
        best_marginal_subset = find_best_in_segment(subsets[i * RATIO:i * RATIO + RATIO], uncovered_elements)
        solution.append(best_marginal_subset)
        uncovered_elements = uncovered_elements - set(best_marginal_subset)
    return solution, uncovered_elements


def main():
    print('helllooo')
    subsets = make_universe()
    print('This is the universe: ' + str(subsets))
    results_sum = 0.0
    for i in range(TEST_LOOP):
        subsets = random.sample(subsets, NUMBER_OF_SUBSETS)
        print('This is the universe after sampling: ' + str(subsets))
        solution, uncovered_elements = proposed_algorithm(subsets)
        results_sum += (1 - float(len(uncovered_elements) / SIZE_OF_UNIVERSE))

    print('The expectation of the competetive ratio is %f' % float(results_sum / TEST_LOOP))


if __name__ == '__main__':
    main()
