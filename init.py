import os


def init():
    if not os.path.exists('dcms'):
        os.mkdir('dcms')
    
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('output'):
        os.mkdir('output')
    
    if not os.path.exists('slices'):
        os.mkdir('slices')

if __name__ == '__main__':
    init()