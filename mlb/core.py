import datetime
from ipdb import post_mortem, set_trace
import ast
from itertools import zip_longest
from importlib import reload

from mlb.exception import format_exception
from mlb.color import *


import numpy as np
import inspect
import subprocess as sp
import os
import sys
import select
from bdb import BdbQuit
from contextlib import contextmanager
import traceback as tb
import time
from math import ceil


def die(s):
    raise Exception(mk_red(f"Error:{s}"))


def warn(s):
    yellow(f"WARN:{s}")

# cause nobody likes ugly paths


def pretty_path(p):
    return p.replace(homedir, '~')

# returns ['util','repl','main',...] These are the names of the source modules.
# this is used in reload_modules()


def module_ls():
    files = os.listdir(src_path)
    files = list(filter(lambda x: x[-3:] == '.py', files))  # only py files
    mod_names = [x[:-3] for x in files]  # cut off extension
    return mod_names

# takes the result of sys.modules as an argument
# 'verbose' will cause the unformatted exception to be output as well
# TODO passing in sys.modules may be unnecessary bc util.py may share the same sys.modules
# as everything else. Worth checking.


def reload_modules(mods_dict, verbose=False):
    failed_mods = []
    for mod in module_ls():
        if mod in mods_dict:
            try:
                reload(mods_dict[mod])
                #blue('reloaded '+mod)
            except Exception as e:
                failed_mods.append(mod)
                print(format_exception(e, src_path,
                                       ignore_outermost=1, verbose=verbose))
                pass
    return failed_mods  # this is TRUTHY if any failed


# a version of zip that forces things to be equal length
def zip_equal(*iterables):
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if any([sentinel is x for x in combo]):
            raise ValueError('Iterables have different lengths')
        yield combo


# allows you to toggle whether mlb.log() will print
def get_verbose():
    return 'MLB_VERBOSE' in os.environ and os.environ['MLB_VERBOSE'] == '1'
def set_verbose():
    os.environ['MLB_VERBOSE'] = '1'
def log(*args,**kwargs):
    if 'MLB_VERBOSE' in os.environ and os.environ['MLB_VERBOSE'] == '1':
        print(*args,**kwargs)



class ProgressBar:
    def __init__(self, num_steps, num_dots=10):
        self.num_steps = num_steps
        self.num_dots = num_dots
        self.curr_step = 0
        self.dots_printed = 0

    def step(self):
        expected_dots = ceil(self.curr_step/self.num_steps*self.num_dots)
        dots_to_print = expected_dots - self.dots_printed
        if dots_to_print > 0:
            print('.'*dots_to_print, end='', flush=True)
        self.dots_printed = expected_dots
        self.curr_step += 1
        if self.curr_step == self.num_steps:
            print('!\n', end='', flush=True)


callback_inputs = []


def callback(keyword, fn):
    assert callable(fn)
    if len(callback_inputs) > 0:
        if keyword in callback_inputs:
            callback_inputs.remove(keyword)
            return fn()
    while select.select([sys.stdin, ], [], [], 0.0)[0]:
        try:
            line = input().strip()
        except EOFError:
            return
        if line.strip() == keyword:
            return fn()
        else:
            callback_inputs.append(line.strip())

def freezer(keyword='pause'):
    return callback(keyword, set_trace)
    
def predicate(keyword):
    return callback(keyword, lambda:True)

@contextmanager
def debug(debug=True, ctrlc=None, crash=None):
    """
    `debug`: if False, don't actually do anything
    `ctrlc`: a lambda that will be called on KeyboardInterrupt right before sys.exit(1)
        Note we suppress the usual traceback because those generally aren't desirable for keyboard interrupts.
    `crash`: a lambda that will be called right before entering the postmortem debugger
        Note that `raise` will always happen after this
    """
    if not debug:
        yield None
        return

    try:
        yield None
    except BdbQuit: # someone entered a debugger within the program then used ctrl-d to quit it (i think)
        raise
    except KeyboardInterrupt:
        magenta("[KeyboardInterrupt]")
        if ctrlc is not None:
            ctrlc()
        sys.exit(1)
    except Exception as e:
        red("[CRASH]")
        print(''.join(tb.format_exception(e.__class__, e, e.__traceback__)))
        print(format_exception(e, ''))
        print("curr time:",datetime.datetime.now())
        if crash is not None:
            crash()
        post_mortem()
        sys.exit(1)


# unique timers by name
_timers = {}


def get_timer(name, **kwargs):
    if name not in _timers:
        _timers[name] = Timer(**kwargs)
    return _timers[name]


class Time():
    def __init__(self, name, parent, cumulative=False):
        self.name = name
        self.cumulative = cumulative
        self.count = 0
        self._start = None
        self.elapsed = 0
        self.avg = None
        self.parent = parent

    def start(self):
        self._start = time.time()

    def stop(self):
        self.count += 1
        dt = time.time() - self._start
        self.elapsed = (self.elapsed + dt) if self.cumulative else dt
        if self.cumulative:
            self.avg = self.elapsed/self.count

    def percent(self):
        assert 'total' in self.parent.timers
        if self.parent.timers['total'].elapsed == 0:
            assert self.elapsed == 0
            return 0
        return self.elapsed/self.parent.timers['total'].elapsed*100

    def __repr__(self):
        if self.name != 'total' and 'total' in self.parent.timers and self.parent.timers['total'].elapsed != 0:
            percent = self.percent()
            return f'{self.name+":":<{self.parent.longest_name+1}} tot:{str(self.elapsed)+",":<23} avg:{self.avg}, {percent:.3f}%'
        return f'{self.name+":":<{self.parent.longest_name+1}} tot:{str(self.elapsed)+",":<23} avg:{self.avg}'


class Timer:
    def __init__(self, cumulative=True):
        self.timers = {}
        self.most_recent = None
        self.cumulative = cumulative
        self.longest_name = 5

    def start(self, name='timer', cumulative=None):
        if len(name) > self.longest_name:
            self.longest_name = len(name)
        if cumulative is None:
            cumulative = self.cumulative
        self.most_recent = name
        if name not in self.timers:
            self.timers[name] = Time(name, self, cumulative)
        if not hasattr(self, name):
            setattr(self, name, self.timers[name])
        self.timers[name].start()
        return self

    def clear(self):
        for name in self.timers:
            if isinstance(getattr(self, name), Time):
                delattr(self, name)
        self.timers = {}

    def stop(self, name=None):
        """
        Convenience: if you dont provide `name` it uses the last one that `start` was called with. stop() also returns the elapsed time and also does a setattr to set the field with the name `name` to the elapsed time.
        """
        if name is None:
            name = self.most_recent
        assert name in self.timers, f"You need to .start() your timer '{name}'"
        self.timers[name].stop()
        return self.timers[name]

    def print(self, name=None):
        if name is None:
            name = self.most_recent
        print(self.stop(name))

    def __repr__(self):
        body = []
        for timer in self.timers.values():
            body.append(repr(timer))
        return 'Timers:\n'+',\n'.join(body)


_timer = Timer()


def clock(fn):

    tree = ast.parse(inspect.getsource(fn))
    fndef = tree.body[0]
    fndef.name = '_timed_fn'
    fndef.decorator_list = []
    tglobal = ast.parse('mlb._timer = Timer()').body[0]  # Expr
    tstart = ast.parse('mlb._timer.start("4")').body[0]  # Expr
    tstop = ast.parse('mlb._timer.stop("4")').body[0]  # Expr
    body = [tglobal]
    for stmt in fndef.body:
        body.extend([tstart, stmt, tstop])
    fndef.body = body
    ast.fix_missing_locations(tree)

    code = compile(tree, '<string>', 'exec')
    exec(code)
    return locals()['_timed_fn']

    breakpoint()

    lines, def_line = inspect.getsourcelines(fn)
    lines = lines[1:]  # strip away the decorator line
    def_line += 2
    assert lines[0].strip().startswith('def')

    out = []
    timer_obj = f'_timer'
    # `def` header. lstrip to fix indent
    out.append(lines[0].replace(fn.__name__, '_timed_fn').lstrip())

    first = True

    for lineno, line in enumerate(lines[1:]):  # fn body
        lineno += def_line
        nocomment = line[:line.find('#')].strip()
        if nocomment == '' or nocomment.endswith(":"):
            out.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]

        out.append(f'{indent}{timer_obj}.start("{lineno}")')
        out.append(line)
        out.append(f'{indent}{timer_obj}.stop("{lineno}")')

    fn_text = '\n'.join(out)

    global _timer
    _timer = Timer()
    try:
        exec(fn_text)  # define the function
    except:
        print(fn_text)
        raise
    x = locals()['_timed_fn']

    def wrapper(*args, **kwargs):
        _timer.start('total')
        ret = x(*args, **kwargs)
        _timer.stop('total')
        return ret

    wrapper.dt = _timer
    return wrapper

