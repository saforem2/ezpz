"""
ezpz/utils/yeet_env.py

Utility to transfer and extract a tarball across distributed nodes using ezpz.
"""
import sys

def main():
    import ezpz
    import ezpz.pbs
    import ezpz.launch

    logger = ezpz.get_logger(__name__)

    _ = ezpz.setup_torch()
    logger.info(
        f"Total number of ranks (one per node): {ezpz.get_world_size()}"
    )
    return ezpz.launch.launch(
        launch_cmd=ezpz.pbs.get_pbs_launch_cmd(ngpu_per_host=1),
        cmd_to_launch=" ".join([
            "python3",
            "-m",
            "ezpz.utils.yeet_tarball",
            " ".join(sys.argv[1:]),
        ])
        #     "--src=/flare/datascience/foremans/2025-08-pt29.tar.gz",
        #     "--dst=/tmp/pt29.tar.gz",
        #     "--d",
        # ])
    )

if __name__ == "__main__":
    main()
