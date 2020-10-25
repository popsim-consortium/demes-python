import sys

import demes


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} file.yml", file=sys.stderr)
        exit(1)
    demes.load(sys.argv[1])


if __name__ == "__main__":
    main()
