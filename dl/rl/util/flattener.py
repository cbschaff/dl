from collections import OrderedDict

class Flattener(object):
    """
    Flattens nested lists, tuples, and dictionaries to a list and remembers
    structure so that the list can be unflattened later.
    """
    def __init__(self):
        self.structure = None

    def get_structure(self, stuff):
        self.structure = self._get_structure(stuff)

    def _get_structure(self, stuff):
        if isinstance(stuff, (list, tuple)):
            struct = []
            for t in stuff:
                s = self._get_structure(t)
                struct.append(s)
            return struct

        elif isinstance(stuff, dict):
            struct = OrderedDict()
            for k in stuff:
                s = self._get_structure(stuff[k])
                struct[k] = s
            return struct

        else:
            return None

    def tolist(self, stuff):
        return self._tolist(stuff, self.structure)

    def _tolist(self, stuff, struct):
        out = []
        if isinstance(struct, (list, tuple)):
            for i,s in enumerate(struct):
                out.extend(self._tolist(stuff[i], s))
        elif isinstance(struct, OrderedDict):
            for k,s in struct.items():
                out.extend(self._tolist(stuff[k], s))
        else:
            out = [stuff]
        return out

    def fromlist(self, l):
        assert isinstance(l, (list, tuple)), "Input must be a list."
        out, nused = self._fromlist(l, self.structure)
        assert nused == len(l), "Incorrect unflattening structure"
        return out

    def _fromlist(self, l, struct):
        if isinstance(struct, (list, tuple)):
            ind, out = 0, []
            out = []
            for s in struct:
                t, nused = self._fromlist(l[ind:], s)
                out.append(t)
                ind += nused
            return out, ind
        elif isinstance(struct, OrderedDict):
            ind, out = 0, {}
            for k,s in struct.items():
                t, nused = self._fromlist(l[ind:], s)
                out[k] = t
                ind += nused
            return out, ind
        else:
            return l[0], 1



if __name__ == '__main__':
    import unittest

    class TestFlattener(unittest.TestCase):
        def test(self):
            stuff = [{1:3, 'hello': 2}, 'stuff', [1,2,'bob',{'h':2, 1:5}]]
            f = Flattener()
            f.get_structure(stuff)
            l = f.tolist(stuff)
            stuff2 = f.fromlist(l)

            assert stuff2 == stuff
            try:
                f.fromlist(l[1:])
                assert False
            except:
                pass


    unittest.main()
