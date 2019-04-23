import torch
import torch.nn as nn
import torch.nn.functional as F


class Trial:
    def __init__(self, mod, shape, silent=False):
        self.t = torch.zeros(shape)
        self.mod = mod
        self.shapes = {}
        self.silent = silent
        self.original_shape = self.t.shape

# __getattr__ deprecated because of unintuitive interface
#    def __getattr__(self,name):
#        if not hasattr(self.t,name):
#            raise AttributeError
#        attr = getattr(self.t,name)
#        if not callable(attr): # if callable lets wrap it in a fancy closure that assigns self.t to the result when someone calls it
#            return attr # non-callable attr
#        def wrapper(*args,**kwargs):
#            old_shape = self.t.shape
#            res = attr(*args,**kwargs)
#            if isinstance(res,torch.Tensor):
#                self.t = res # overwrite our old tensor if the function returned a tensor
#            self.print("{}: {} -> {}".format(attr.__name__, tuple(old_shape), tuple(self.t.shape)))
#            if not isinstance(res,torch.Tensor):
#                return res #if the function didn't return a tensor, return whatever it returned
#            return self # if it did return a tensor, return our Trial instance. Now trial = trial.view() will work.
#        return wrapper

    def reset(self):
        self.t = torch.zeros(self.original_shape)

    #tr_idx is None if model.forward() returns a tensor, 0 if it returns a tuple with the first element being the desired tensor, etc.
    def forward(self, reset=True, assert_same=False, tr_idx=None):
        self.print("Running full forward pass...")
        shape_before_reset = self.t.shape
        if reset:
            self.reset()
        old_shape = self.t.shape
        with torch.no_grad():
            res = self.mod.forward(self.t)
            if tr_idx is None:
                self.t = res
            else:
                self.t = res[tr_idx]
        self.print("forward(): {} -> {}".format(tuple(old_shape),tuple(self.t.shape)))
        if assert_same:
            assert(shape_before_reset == self.t.shape)

    def print(self,msg):
        if not self.silent:
            print(msg)

    def new_shape(self, shape, reason=None):
        old_shape = self.t.shape
        self.t = torch.zeros(shape)
        if reason is None:
            reason = 'manual trial.new_shape'
        self.print("{}: {} -> {}".format(reason,tuple(old_shape),tuple(self.t.shape)))

# convenience function
    def view(self, shape):
        old_shape = self.t.shape
        self.t = self.t.view(shape)
        self.print("view: {} -> {}".format(tuple(old_shape),tuple(self.t.shape)))

    def clone(self):
        return Trial(self.mod, self.t.shape, silent=self.silent)

    # usage:
    # shape = trial.shape()
    # batches = trial.shape(0)
    def shape(self,dim=-1):
        if dim==-1:
            return self.t.shape
        else:
            return self.t.shape[dim]

    # usage:
    # self.flatten = trial.flat_shape()
    # trial.view(self.flatten)
    # converts NABCDE to NF where F=A*B*C*D*E
    def flat_shape(self):
        flat = 1
        for dimsize in self.t.shape[1:]:
            flat = flat*dimsize
        return (-1,flat)

    # note if you wanted to do a method like x.view that didn't have a trial.____ equivalent you could do either:
    #   trial.t = trial.t.view(...)
    # or
    #   trial.apply(lambda t: t.view(...))
    # or
    #   trial.apply(torch.Tensor.view,...)
    def apply(self, callable_obj, *args, name=None, depth=0, tr_idx=None, **kwargs): # note this works with functions not just layers

        temp = self.clone() # for use in nn.Sequential case
        old_shape = self.t.shape
        with torch.no_grad():
            res =  callable_obj(self.t,*args,**kwargs)
            if tr_idx is not None:
                res = res[tr_idx] # do a deref to get the tensor
            self.t = res


        if isinstance(callable_obj,nn.ReLU):
            return self.t.shape # abort early. you dont want this printed at all, really.

        # this assert is good to have. Programs will crash quickly during __init__ and people will correct their mistakes. Much better than someone accidently using apply when they meant apply_noassign and having to track down the bug.
        assert isinstance(self.t,torch.Tensor), "you should be using apply_noassign. The function ({}) you used in apply() returns a nontensor".format(callable_obj)

        if name is not None:
            self.shapes[name] = self.t.shape # also store in shapes[name] if a name is given
        self.shapes[callable_obj] = self.t.shape

        if name is None: # now lets create a nice default name
            if '__name__' in dir(callable_obj):
                name = callable_obj.__name__
            else:
                name = str(callable_obj)
                name = name[:name.find('(')] # to pretty up the layers that look like Linear(tons of stuff ...)

        if self.t.shape != old_shape:
            self.print("{}{}: {} -> {}".format('\t'*depth, name, tuple(old_shape), tuple(self.t.shape)))
        else:
            self.print("{}{} (no effect)".format('\t'*depth, name))

        if isinstance(callable_obj,nn.Sequential):
            for layer in callable_obj:
                temp.apply(layer, depth=depth+1)
        return self.t.shape

    def apply_noassign(self, callable_obj):
        return callable_obj(self.t)





