"""
ezpz/test.py
"""
from ezpz import setup_torch, setup_tensorflow


def test(
        framework: str = 'pytorch',
        backend: str = 'deepspeed',
        port: int | str = '5432'
):
    if framework == 'pytorch':
        _ = setup_torch(
            backend=backend,
            port=port,
        )
    elif framework == 'tensorflow':
        _ = setup_tensorflow()
    else:
        raise ValueError
    # WORLD_SIZE = get_world_size()
    # print(f'{RANK} / {WORLD_SIZE}')


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
    test(framework=framework, backend=backend, port=port)
