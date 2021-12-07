# stolen from tskit
import re

SUBS = [
    (r"{user}`([A-Za-z0-9-]*)`", r"[@\1](https://github.com/\1)"),
    (
        r"{pr}`([0-9]*)`",
        r"[#\1](https://github.com/popsim-consortium/demes-python/issues/\1)",
    ),
    (
        r"{issue}`([0-9]*)`",
        r"[#\1](https://github.com/popsim-consortium/demes-python/issues/\1)",
    ),
]


def process_log(log):
    delimiters_seen = 1
    for line in log:
        if line.startswith("##"):
            delimiters_seen += 1
            continue
        if delimiters_seen == 3:
            return
        if delimiters_seen % 2 == 0:
            for pattern, replace in SUBS:
                line = re.sub(pattern, replace, line)
            yield line


with open("CHANGELOG.md") as f:
    print("".join(process_log(f.readlines())))
