from ezpz.launch import launch


def main():
    import sys

    test_script = f"{sys.executable} -m ezpz.test_dist"
    cmd_args = None
    if len(sys.argv) > 1:
        cmd_args = " ".join(sys.argv[1:])
        if cmd_args.split(" ")[0] in ["python", "python3"]:
            cmd_args = f"{sys.executable} {cmd_args}"
    if cmd_args is not None:
        test_script = f"{test_script} {cmd_args}"
    return launch(cmd_to_launch=test_script)


if __name__ == "__main__":
    main()
