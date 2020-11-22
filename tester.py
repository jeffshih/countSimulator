import weakref
#from Util import *
from detection import point

class InsaneClass(object):
    _alive = []
    id = 0
    def __new__(cls):
        cls.id +=1
        self = super().__new__(cls)
        InsaneClass._alive.append(self)
        return weakref.proxy(self)

    def commit_suicide(self):
        self._alive.remove(self)


if __name__ == "__main__":
    instance = InsaneClass()
    print(instance.id)
    instance2 = InsaneClass()
    print(instance2.id)
    #instance.commit_suicide()
    #print(instance)
    A = point(10,12)
    print(A)
    print(A[0])
    print(A[3])