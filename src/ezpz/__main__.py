"""
ezpz/__main__.py

"""
from ezpz.dist import check


if __name__ == '__main__':
    import sys
    try:
        framework = sys.argv[1]
    except IndexError:
        framework = 'pytorch'
    try:
        backend = sys.argv[2]
    except IndexError:
        backend = 'deepspeed'
    try:
        port = sys.argv[3]
    except IndexError:
        port = '5432'
    check(framework=framework, backend=backend, port=port)
