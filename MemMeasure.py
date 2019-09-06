from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
from keras.models import load_model
import argparse
import QueryRecommender as QR
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


##### Example call #####

def computeMemoryReq(args):
    for fileName in eval(args.list):
        obj = QR.readFromPickleFile(fileName)
        print(total_size(obj, verbose=True))
    if args.model is not None:
        modelRNN = load_model(args.model)
        print(total_size(modelRNN, verbose=True))

if __name__ == '__main__':
    #d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
    parser = argparse.ArgumentParser()
    parser.add_argument("-list", help="list of paths to tuple/list/deque/dict/set/frozenset obj", type=str, required=True)
    parser.add_argument("-model", help="path to RNN model filename to analyze", type=str, required=False)
    # parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
    computeMemoryReq(args)
    print(total_size(d, verbose=True))
