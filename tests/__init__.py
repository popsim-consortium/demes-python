import math

import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st

import demes


@st.composite
def deme_names(draw, max_length=20):
    name = draw(st.text(min_size=1, max_size=max_length))
    # Names must be valid Python identifiers.
    hyp.assume(name.isidentifier())
    return name


@st.composite
def yaml_strings(draw, min_size=1, max_size=100):
    """
    From https://yaml.org/spec/1.2/spec.html#id2770814

        To ensure readability, YAML streams use only the printable subset of
        the Unicode character set. The allowed character range explicitly
        excludes the C0 control block #x0-#x1F (except for TAB #x9, LF #xA,
        and CR #xD which are allowed), DEL #x7F, the C1 control block #x80-#x9F
        (except for NEL #x85 which is allowed), the surrogate block #xD800-#xDFFF,
        #xFFFE, and #xFFFF.

        On input, a YAML processor must accept all Unicode characters except
        those explicitly excluded above.

        On output, a YAML processor must only produce acceptable characters.
        Any excluded characters must be presented using escape sequences.
        In addition, any allowed characters known to be non-printable should
        also be escaped. This isn’t mandatory since a full implementation would
        require extensive character property tables.
    """
    return draw(
        st.text(
            alphabet=st.characters(
                blacklist_categories=(
                    "Cc",  # control block (C0 and C1)
                    "Cs",  # surrogate block
                ),
                blacklist_characters=("\ufffe", "\uffff"),
                whitelist_characters=("\x09", "\x0a", "\x0d", "\x85"),
            ),
            min_size=min_size,
            max_size=max_size,
        )
    )


@st.composite
def epochs_lists(draw, start_time=math.inf, max_epochs=5):
    """
    A hypothesis strategy for creating lists of Epochs for a deme.

    .. code-block::

        @hypothesis.given(epoch_lists())
        test_something(self, epoch_list):
            # epoch_list has type ``list of Epoch``
            pass

    :param float start_time: The start time of the deme.
    :param int max_epochs: The maximum number of epochs in the list.
    """
    assert max_epochs >= 2
    times = draw(
        st.lists(
            st.floats(min_value=0, max_value=start_time, exclude_max=True),
            unique=True,
            min_size=1,
            max_size=max_epochs,
        )
    )
    times.sort(reverse=True)
    epochs = []

    for i, end_time in enumerate(times):
        start_size = draw(
            st.floats(min_value=0, exclude_min=True, allow_infinity=False)
        )
        if i == 0 and math.isinf(start_time):
            end_size = start_size
        else:
            end_size = draw(
                st.floats(min_value=0, exclude_min=True, allow_infinity=False)
            )
        cloning_rate = draw(st.floats(min_value=0, max_value=1))
        selfing_rate = draw(st.floats(min_value=0, max_value=1 - cloning_rate))

        epochs.append(
            dict(
                end_time=end_time,
                start_size=start_size,
                end_size=end_size,
                cloning_rate=cloning_rate,
                selfing_rate=selfing_rate,
            )
        )

    return epochs


@st.composite
def graphs(draw, max_demes=5, max_interactions=5, max_epochs=5):
    """
    A hypothesis strategy for creating a Graph.

    .. code-block::

        @hypothesis.given(graphs())
        def test_something(self, g):
            # g has type ``Graph``
            pass

    :param int max_demes: The maximum number of demes in the graph.
    :param int max_interactions: The maximum number of migrations plus pulses
        in the graph.
    :param int max_epochs: The maximum number of epochs per deme.
    """
    generation_time = draw(st.none() | st.floats(min_value=1e-9, max_value=1e6))
    if generation_time is None:
        time_units = "generations"
    else:
        time_units = draw(yaml_strings())
    b = demes.Builder(
        description=draw(yaml_strings()),
        generation_time=generation_time,
        time_units=time_units,
        doi=draw(st.lists(yaml_strings(), max_size=3)),
    )

    for deme_name in draw(st.sets(deme_names(), min_size=1, max_size=max_demes)):
        ancestors = []
        proportions = []
        start_time = math.inf
        n_demes = 0 if "demes" not in b.data else len(b.data["demes"])
        if n_demes > 0:
            # draw indices into demes list to use as ancestors
            anc_idx = draw(
                st.lists(
                    st.integers(min_value=0, max_value=n_demes - 1),
                    unique=True,
                    max_size=n_demes,
                )
            )
            if len(anc_idx) > 0:
                time_hi = min(b.data["demes"][j]["start_time"] for j in anc_idx)
                time_lo = max(
                    b.data["demes"][j]["epochs"][-1]["end_time"] for j in anc_idx
                )
                # If time_hi > time_lo, the proposed ancestors exist
                # at the same time. So we draw a number for the deme's
                # start_time, which must be in the half-open interval
                # [time_lo, time_hi), with the further constraint that the
                # start_time cannot be 0.
                # However, there may not be any floating point numbers between
                # 0 and time_hi even if time_hi > 0, so we check that time_hi
                # is greater than the smallest positive number.
                if (time_lo > 0 and time_hi > time_lo) or (
                    time_lo == 0 and time_hi > np.finfo(float).tiny
                ):
                    # Draw a start time and the ancestry proportions.
                    start_time = draw(
                        st.floats(
                            min_value=time_lo,
                            max_value=time_hi,
                            exclude_max=True,
                            # Can't have start_time=0.
                            exclude_min=time_lo == 0,
                        )
                    )
                    ancestors = [b.data["demes"][j]["name"] for j in anc_idx]
                    if len(ancestors) == 1:
                        proportions = [1.0]
                    else:
                        proportions = draw(
                            st.lists(
                                st.integers(min_value=1, max_value=10 ** 9),
                                min_size=len(ancestors),
                                max_size=len(ancestors),
                            )
                        )
                        psum = sum(proportions)
                        proportions = [p / psum for p in proportions]
        b.add_deme(
            name=deme_name,
            description=draw(st.none() | yaml_strings()),
            ancestors=ancestors,
            proportions=proportions,
            epochs=draw(epochs_lists(start_time=start_time, max_epochs=max_epochs)),
            start_time=start_time,
        )

    n_interactions = draw(st.integers(min_value=0, max_value=max_interactions))
    n_tries = 100
    n_demes = len(b.data["demes"])
    for j in range(n_demes - 1):
        for k in range(j + 1, n_demes):
            dj = b.data["demes"][j]["name"]
            dk = b.data["demes"][k]["name"]
            time_lo = max(
                b.data["demes"][j]["epochs"][-1]["end_time"],
                b.data["demes"][k]["epochs"][-1]["end_time"],
            )
            time_hi = min(
                b.data["demes"][j]["start_time"], b.data["demes"][k]["start_time"]
            )
            # Draw asymmetric migrations.
            #
            # If time_hi > time_lo, then demes j and k exist at the same time
            # during the half-open interval [time_lo, time_hi).
            #
            # We wish to draw a migration start_time and end_time on this
            # interval, and in the worst case (smallest interval) we will
            # draw start_time=time_hi, end_time=time_lo, which is valid.
            if time_hi <= time_lo:
                continue
            n = draw(st.integers(min_value=0, max_value=n_interactions))
            successes = 0
            migration_intervals = {(dj, dk): [], (dk, dj): []}
            try_i = 0
            while successes < n and try_i < n_tries:
                try_i += 1
                source, dest = dj, dk
                if draw(st.booleans()):
                    source, dest = dk, dj
                times = draw(
                    st.lists(
                        st.floats(min_value=time_lo, max_value=time_hi),
                        unique=True,
                        min_size=2,
                        max_size=2,
                    )
                )
                no_overlap = True
                for existing_interval in migration_intervals[(dj, dk)]:
                    # check that our drawn interval doesn't overlap with existing mig
                    if not (
                        min(times) >= existing_interval[0]
                        or max(times) <= existing_interval[1]
                    ):
                        no_overlap = False
                        break
                if no_overlap:
                    migration_intervals[(dj, dk)].append([max(times), min(times)])
                    successes += 1
                    b.add_migration(
                        source=source,
                        dest=dest,
                        start_time=max(times),
                        end_time=min(times),
                        rate=draw(
                            st.floats(min_value=0, max_value=1, exclude_max=True)
                        ),
                    )
            n_interactions -= successes
            if n_interactions <= 0:
                break

            # Draw pulses.
            #
            # We wish to draw a time for the pulse. This must be in the open
            # interval (time_lo, time_hi) to ensure the pulse doesn't happen
            # at any deme's start_time or end_time, which would be invalid.
            # So we check there is at least one floating point number between
            # time_lo and time_hi.
            if time_hi <= np.nextafter(time_lo, np.inf, dtype=float):
                continue
            n = draw(st.integers(min_value=0, max_value=n_interactions))
            n_interactions -= n
            for _ in range(n):
                source, dest = dj, dk
                if draw(st.booleans()):
                    source, dest = dk, dj
                time = draw(
                    st.floats(
                        min_value=time_lo,
                        max_value=time_hi,
                        exclude_min=True,
                        exclude_max=True,
                    )
                )
                b.add_pulse(
                    source=source,
                    dest=dest,
                    time=time,
                    proportion=draw(
                        st.floats(
                            min_value=0, max_value=1, exclude_min=True, exclude_max=True
                        )
                    ),
                )
            if n_interactions <= 0:
                break
        if n_interactions <= 0:
            break

    return b.resolve()
