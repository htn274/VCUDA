from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default= 'linear.yml',
                            help='Config path for running experiment')
    parser.add_argument('--methods', type=str, nargs='+', default=['VCUDA', 'GraNDAG', 'MCSL', 'DiBS', 'DDS',])
    args = parser.parse_args() 