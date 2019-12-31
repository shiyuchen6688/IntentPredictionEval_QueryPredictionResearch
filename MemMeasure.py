from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
from keras.models import load_model
import ReverseEnggQueries_selOpConst
import argparse
import ParseConfigFile as parseConfig
from ParseConfigFile import getConfig
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
    if args.list is not None:
        fileList = args.list.split(";")
        for fileName in fileList:
            fileName = fileName.replace("\"","")
            obj = QR.readFromPickleFile(fileName)
            fileSize = total_size(obj, verbose=False)
            print("Size of "+fileName+" is "+str(fileSize))
    if args.model is not None:
        modelRNN = load_model(args.model)
        fileSize = total_size(modelRNN, verbose=False)
        print("Size of " + args.model + " is " + str(fileSize))
        configDict = parseConfig.parseConfigFile(args.config)
        schemaDicts = ReverseEnggQueries_selOpConst.readSchemaDicts(configDict)
        schemaDictSize = total_size(schemaDicts, verbose=False)
        print("Size of schemaDicts is " + str(schemaDictSize))
    return

def computeMemQueries(args):
    configDict = parseConfig.parseConfigFile(args.config)
    intentSessionFile = QR.fetchIntentFileFromConfigDict(configDict)
    sessionStreamDict = {}
    with open(intentSessionFile) as f:
        for line in f:
            (sessID, queryID, curQueryIntent, sessionStreamDict) = QR.updateSessionDict(line, configDict,
                                                                                        sessionStreamDict)
    f.close()
    size = total_size(sessionStreamDict, verbose=False)
    print("Size of sessionStreamDict is: "+size)

if __name__ == '__main__':
    #d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", help="path to config dictionary", type=str, required=True)
    parser.add_argument("-list", help="semicolon separated list of paths to tuple/list/deque/dict/set/frozenset obj", type=str, required=False)
    parser.add_argument("-model", help="path to RNN model filename to analyze", type=str, required=False)
    # parser.add_argument("-lineNum", help="line Number to analyze", type=int, required=True)
    args = parser.parse_args()
   # computeMemoryReq(args)
    computeMemQueries(args)
