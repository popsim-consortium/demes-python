import textwrap

import demes


def to_demes(deme_graph: demes.DemeGraph):
    def str_fix(string, indent=0, width=80):
        """
        Add quotes around string, and wrap long strings.
        """
        if not isinstance(string, str):
            return string
        if len(string) > width - indent:
            indent2 = indent + 4
            lines = textwrap.wrap(string, width=width - indent2)
            string = (' "\n' + indent2 * " " + '"').join(lines)
            string = "(\n" + indent2 * " " + f'"{string}"\n' + indent * " " + ")"
        else:
            string = f'"{string}"'
        return string

    d = deme_graph.asdict_compact()
    py_str = []
    py_str.append("g = demes.DemeGraph(")
    for key in [
        "description",
        "time_units",
        "generation_time",
        "doi",
        "selfing_rate",
        "cloning_rate",
    ]:
        val = d.get(key)
        if val is not None:
            val = str_fix(val, indent=4)
            py_str.append(f"    {key}={val},")
    py_str[-1] = py_str[-1][:-1]  # remove trailing comma
    py_str.append(")")

    for deme_id, deme_attr in d["demes"].items():
        py_str.append("g.deme(")
        py_str.append(f'    id="{deme_id}",')
        for key in [
            "description",
            "ancestors",
            "proportions",
            "start_time",
            "end_time",
            "initial_size",
            "final_size",
            "selfing_rate",
            "cloning_rate",
        ]:
            val = deme_attr.get(key)
            if val is not None:
                val = str_fix(val, indent=4)
                py_str.append(f"    {key}={val},")

        epochs = deme_attr.get("epochs")
        if epochs is not None:
            py_str.append("    epochs=[")
            for epoch in epochs:
                py_str.append(8 * " " + "Epoch(")
                for key in [
                    "start_time",
                    "end_time",
                    "initial_size",
                    "final_size",
                    "size_function",
                    "selfing_rate",
                    "cloning_rate",
                ]:
                    val = epoch.get(key)
                    if val is not None:
                        if isinstance(val, str):
                            val = f'"{val}"'
                        py_str.append(12 * " " + f"{key}={val},")
                py_str[-1] = py_str[-1][:-1]  # remove trailing comma
                py_str.append("        ),")
            py_str[-1] = py_str[-1][:-1]  # remove trailing comma
            py_str.append("    ],")

        py_str[-1] = py_str[-1][:-1]  # remove trailing comma
        py_str.append(")")

    migrations = d.get("migrations")
    if migrations is not None:
        symmetric = migrations.get("symmetric", [])
        for migration in symmetric:
            py_str.append("g.symmetric_migration(")
            for key in [
                "demes",
                "start_time",
                "end_time",
                "rate",
            ]:
                val = migration.get(key)
                if val is not None:
                    if isinstance(val, str):
                        val = f'"{val}"'
                    py_str.append(f"    {key}={val},")
            py_str[-1] = py_str[-1][:-1]  # remove trailing comma
            py_str.append(")")

        asymmetric = migrations.get("asymmetric", [])
        for migration in asymmetric:
            py_str.append("g.migration(")
            for key in [
                "source",
                "dest",
                "start_time",
                "end_time",
                "rate",
            ]:
                val = migration.get(key)
                if val is not None:
                    if isinstance(val, str):
                        val = f'"{val}"'
                    py_str.append(f"    {key}={val},")
            py_str[-1] = py_str[-1][:-1]  # remove trailing comma
            py_str.append(")")

    pulses = d.get("pulses", [])
    for pulse in pulses:
        py_str.append("g.pulse(")
        for key in [
            "source",
            "dest",
            "time",
            "proportion",
        ]:
            val = pulse.get(key)
            if val is not None:
                if isinstance(val, str):
                    val = f'"{val}"'
                py_str.append(f"    {key}={val},")
        py_str[-1] = py_str[-1][:-1]  # remove trailing comma
        py_str.append(")")

    return "\n".join(py_str)

    py_str = "\n".join(py_str)
    exec(py_str)
    assert deme_graph.isclose(g)
    return py_str


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: {sys.argv[0]} file.yml")
        exit(1)
    g = demes.load(sys.argv[1])
    print(to_demes(g))
