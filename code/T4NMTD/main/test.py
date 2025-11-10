
from env.GetEnv import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', type=str, default='log1/task1/task1.csv', help='log path')
    parser.add_argument('-i', type=str, default='inst11', help='inst name')
    parser.add_argument('-r', type=str, default='waterworld1', help='inst name')
    parser.add_argument('-o', type=int, default=5, help='process num')
    parser.add_argument('-t', type=int, default=3000, help='training time')
    args = parser.parse_args()

    training_time = args.t
    option_num = args.o
    upper_domain = 'high_level_benchmarks/waterworld/' + args.r + '.rddl'
    lower_domain = 'low_level_benchmarks/waterworld/' + args.r + '.rddl'
    instance = 'low_level_benchmarks/waterworld/' + args.i + '.rddl'

    name = args.r

    eval_env = GetLowerEnv(name='waterworld1-1')
    print(eval_env.action_space)
    print(eval_env.observation_space)
