import math

import hypothesis as hyp
import hypothesis.strategies as st

import demes


@st.composite
def deme_ids(draw, max_length=20):
    id = draw(st.text(min_size=1, max_size=max_length))
    # IDs must be valid Python identifiers.
    hyp.assume(id.isidentifier())
    return id


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
            st.floats(min_value=0, max_value=start_time),
            unique=True,
            min_size=2,
            max_size=max_epochs,
        )
    )
    times.sort(reverse=True)
    if start_time != times[0]:
        times.insert(0, start_time)
    epochs = []

    for end_time in times[1:]:
        initial_size = draw(
            st.floats(min_value=0, exclude_min=True, allow_infinity=False)
        )
        if math.isinf(start_time):
            final_size = initial_size
        else:
            final_size = draw(
                st.floats(min_value=0, exclude_min=True, allow_infinity=False)
            )
        cloning_rate = draw(st.floats(min_value=0, max_value=1))
        selfing_rate = draw(st.floats(min_value=0, max_value=1))

        epochs.append(
            demes.Epoch(
                start_time=start_time,
                end_time=end_time,
                initial_size=initial_size,
                final_size=final_size,
                cloning_rate=cloning_rate,
                selfing_rate=selfing_rate,
            )
        )
        start_time = end_time

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
    g = demes.Graph(
        description=draw(yaml_strings()),
        generation_time=generation_time,
        time_units=time_units,
        doi=draw(st.lists(yaml_strings(), max_size=3)),
    )

    for id in draw(st.sets(deme_ids(), min_size=1, max_size=max_demes)):
        ancestors = []
        proportions = []
        start_time = math.inf
        if len(g.demes) > 0:
            # draw indices into demes list to use as ancestors
            anc_idx = draw(
                st.lists(
                    st.integers(min_value=0, max_value=len(g.demes) - 1),
                    unique=True,
                    max_size=len(g.demes),
                )
            )
            if len(anc_idx) > 0:
                time_hi = min(g.demes[j].start_time for j in anc_idx)
                time_lo = max(g.demes[j].end_time for j in anc_idx)
                if time_lo < time_hi and time_lo < 1e308:
                    # The proposed ancestors exist at the same time.
                    # Draw a start time and the ancestry proportions.
                    start_time = draw(
                        st.floats(
                            min_value=time_lo,
                            max_value=time_hi,
                            exclude_max=True,
                        )
                    )
                    ancestors = [g.demes[j].id for j in anc_idx]
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
        g.deme(
            id=id,
            description=draw(st.none() | yaml_strings()),
            ancestors=ancestors,
            proportions=proportions,
            epochs=draw(epochs_lists(start_time=start_time, max_epochs=max_epochs)),
            start_time=start_time,
        )

    n_interactions = draw(st.integers(min_value=0, max_value=max_interactions))
    for j in range(len(g.demes) - 1):
        for k in range(j + 1, len(g.demes)):
            dj = g.demes[j].id
            dk = g.demes[k].id
            time_lo = max(g[dj].end_time, g[dk].end_time)
            time_hi = min(g[dj].start_time, g[dk].start_time)
            if time_hi <= time_lo or time_lo > 1e308:
                # Demes j and k don't exist at the same time.
                # (or time_lo is too close to infinity for floats)
                continue
            # Draw asymmetric migrations.
            n = draw(st.integers(min_value=0, max_value=n_interactions))
            n_interactions -= n
            for _ in range(n):
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
                g.migration(
                    source=source,
                    dest=dest,
                    start_time=max(times),
                    end_time=min(times),
                    rate=draw(st.floats(min_value=0, max_value=1, exclude_max=True)),
                )
            if n_interactions <= 0:
                break
            # Draw pulses.
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
                g.pulse(
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

    return g
