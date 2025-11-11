import cProfile
import pstats
import test


def run_all():
    for i in range(4):
        test.run_solution(i)

cProfile.run('run_all()', 'test.stats')


p = pstats.Stats('test.stats')

p.strip_dirs().sort_stats('cumulative').print_stats(20)