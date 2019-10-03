"""Util for nested lists, tuple, and dictionaries of python objects."""


def get_structure(nest):
    """Return a nest with the same structure, but without the data."""
    if isinstance(nest, (list, tuple)):
        struct = []
        for t in nest:
            struct.append(get_structure(t))
        return struct

    elif isinstance(nest, dict):
        struct = {}
        for k in nest:
            struct[k] = get_structure(nest[k])
        return struct

    else:
        return None


def flatten(nest):
    """Flattens a nest to a list."""
    out = []
    if isinstance(nest, (list, tuple)):
        for x in nest:
            out.extend(flatten(x))
    elif isinstance(nest, dict):
        try:
            sorted_keys = sorted(list(nest.keys()))
        except Exception:
            raise ValueError("The keys of dictionaries in nest must be "
                             "sortable!")
        for k in sorted_keys:
            out.extend(flatten(nest[k]))
    else:
        out = [nest]
    return out


def pack_sequence_as(seq, nest):
    """Packs a list/tuple with the structure of nest."""
    assert isinstance(seq, (list, tuple)), "Input must be a list."
    new_nest, nused = _pack_sequence_as(seq, nest)
    assert nused == len(seq), (
        "nest does not have the same number of elements as seq.")
    return new_nest


def _pack_sequence_as(seq, nest):
    if isinstance(nest, (list, tuple)):
        ind, out = 0, []
        for x in nest:
            new_nest, nused = _pack_sequence_as(seq[ind:], x)
            out.append(new_nest)
            ind += nused
        return out, ind

    elif isinstance(nest, dict):
        ind, out = 0, {}
        try:
            sorted_keys = sorted(list(nest.keys()))
        except Exception:
            raise ValueError("The keys of dictionaries in nest must be "
                             "sortable!")
        for k in sorted_keys:
            new_nest, nused = _pack_sequence_as(seq[ind:], nest[k])
            out[k] = new_nest
            ind += nused
        return out, ind
    else:
        return seq[0], 1


if __name__ == '__main__':
    import unittest

    class TestNest(unittest.TestCase):
        """Test."""

        def test(self):
            """Test."""
            nest = [{1: 3, 2: 2}, 'stuff', [1, 2, 'bob', {'h': 2, 's': 5}]]
            nest_no_data = get_structure(nest)
            seq = flatten(nest)
            nest2 = pack_sequence_as(seq, nest)
            nest3 = pack_sequence_as(seq, nest_no_data)

            assert nest == nest2
            assert nest == nest3
            try:
                nest2 = pack_sequence_as(seq[1:], nest)
                assert False
            except Exception:
                pass

    unittest.main()
