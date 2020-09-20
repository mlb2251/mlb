from mlb.color import *
from mlb.core import *



# UTIL
# import datetime
# from pdb import post_mortem
# from pdb import set_trace
# import ast
# from itertools import zip_longest
# from importlib import reload


# import numpy as np
# import inspect
# import subprocess as sp
# import os
# import sys
# import select
# from bdb import BdbQuit
# from contextlib import contextmanager
# from pdb import post_mortem, set_trace
# import traceback as tb
# import time
# from math import ceil
# homedir = os.environ['HOME']
# src_path = os.path.dirname(os.path.realpath(__file__))+'/'
# #src_path = homedir+'/espresso/src/'
# data_path = os.path.join(homedir, '.espresso/')
# error_path = os.path.join(data_path+'error_handling/')
# #repl_path = data_path+'repl-tmpfiles/'
# #pipe_dir = data_path+'pipes/'
# pwd_file = os.path.join(data_path, '.{}.PWD'.format(os.getpid()))


# a version of zip that forces things to be equal length
# def zip_equal(*iterables):
#     sentinel = object()
#     for combo in zip_longest(*iterables, fillvalue=sentinel):
#         if any([sentinel is x for x in combo]):
#             raise ValueError('Iterables have different lengths')
#         yield combo

# def get_verbose():
#     return 'MLB_VERBOSE' in os.environ and os.environ['MLB_VERBOSE'] == '1'

# def set_verbose():
#     os.environ['MLB_VERBOSE'] = '1'
# def log(*args,**kwargs):
#     if 'MLB_VERBOSE' in os.environ and os.environ['MLB_VERBOSE'] == '1':
#         print(*args,**kwargs)



# # unique timers by name
# _timers = {}


# def get_timer(name, **kwargs):
#     if name not in _timers:
#         _timers[name] = Timer(**kwargs)
#     return _timers[name]


# class Time():
#     def __init__(self, name, parent, cumulative=False):
#         self.name = name
#         self.cumulative = cumulative
#         self.count = 0
#         self._start = None
#         self.elapsed = 0
#         self.avg = None
#         self.parent = parent

#     def start(self):
#         self._start = time.time()

#     def stop(self):
#         self.count += 1
#         dt = time.time() - self._start
#         self.elapsed = (self.elapsed + dt) if self.cumulative else dt
#         if self.cumulative:
#             self.avg = self.elapsed/self.count

#     def percent(self):
#         assert 'total' in self.parent.timers
#         if self.parent.timers['total'].elapsed == 0:
#             assert self.elapsed == 0
#             return 0
#         return self.elapsed/self.parent.timers['total'].elapsed*100

#     def __repr__(self):
#         if self.name != 'total' and 'total' in self.parent.timers and self.parent.timers['total'].elapsed != 0:
#             percent = self.percent()
#             return f'{self.name+":":<{self.parent.longest_name+1}} tot:{str(self.elapsed)+",":<23} avg:{self.avg}, {percent:.3f}%'
#         return f'{self.name+":":<{self.parent.longest_name+1}} tot:{str(self.elapsed)+",":<23} avg:{self.avg}'


# class Timer:
#     def __init__(self, cumulative=True):
#         self.timers = {}
#         self.most_recent = None
#         self.cumulative = cumulative
#         self.longest_name = 5

#     def start(self, name='timer', cumulative=None):
#         if len(name) > self.longest_name:
#             self.longest_name = len(name)
#         if cumulative is None:
#             cumulative = self.cumulative
#         self.most_recent = name
#         if name not in self.timers:
#             self.timers[name] = Time(name, self, cumulative)
#         if not hasattr(self, name):
#             setattr(self, name, self.timers[name])
#         self.timers[name].start()
#         return self

#     def clear(self):
#         for name in self.timers:
#             if isinstance(getattr(self, name), Time):
#                 delattr(self, name)
#         self.timers = {}

#     def stop(self, name=None):
#         """
#         Convenience: if you dont provide `name` it uses the last one that `start` was called with. stop() also returns the elapsed time and also does a setattr to set the field with the name `name` to the elapsed time.
#         """
#         if name is None:
#             name = self.most_recent
#         assert name in self.timers, f"You need to .start() your timer '{name}'"
#         self.timers[name].stop()
#         return self.timers[name]

#     def print(self, name=None):
#         if name is None:
#             name = self.most_recent
#         print(self.stop(name))

#     def __repr__(self):
#         body = []
#         for timer in self.timers.values():
#             body.append(repr(timer))
#         return 'Timers:\n'+',\n'.join(body)


# _timer = Timer()


# def clock(fn):

#     tree = ast.parse(inspect.getsource(fn))
#     fndef = tree.body[0]
#     fndef.name = '_timed_fn'
#     fndef.decorator_list = []
#     tglobal = ast.parse('mlb._timer = Timer()').body[0]  # Expr
#     tstart = ast.parse('mlb._timer.start("4")').body[0]  # Expr
#     tstop = ast.parse('mlb._timer.stop("4")').body[0]  # Expr
#     body = [tglobal]
#     for stmt in fndef.body:
#         body.extend([tstart, stmt, tstop])
#     fndef.body = body
#     ast.fix_missing_locations(tree)

#     code = compile(tree, '<string>', 'exec')
#     exec(code)
#     return locals()['_timed_fn']

#     breakpoint()

#     lines, def_line = inspect.getsourcelines(fn)
#     lines = lines[1:]  # strip away the decorator line
#     def_line += 2
#     assert lines[0].strip().startswith('def')

#     out = []
#     timer_obj = f'_timer'
#     # `def` header. lstrip to fix indent
#     out.append(lines[0].replace(fn.__name__, '_timed_fn').lstrip())

#     first = True

#     for lineno, line in enumerate(lines[1:]):  # fn body
#         lineno += def_line
#         nocomment = line[:line.find('#')].strip()
#         if nocomment == '' or nocomment.endswith(":"):
#             out.append(line)
#             continue
#         indent = line[:len(line) - len(line.lstrip())]

#         out.append(f'{indent}{timer_obj}.start("{lineno}")')
#         out.append(line)
#         out.append(f'{indent}{timer_obj}.stop("{lineno}")')

#     fn_text = '\n'.join(out)

#     global _timer
#     _timer = Timer()
#     try:
#         exec(fn_text)  # define the function
#     except:
#         print(fn_text)
#         raise
#     x = locals()['_timed_fn']

#     def wrapper(*args, **kwargs):
#         _timer.start('total')
#         ret = x(*args, **kwargs)
#         _timer.stop('total')
#         return ret

#     wrapper.dt = _timer
#     return wrapper


# class VisdomPlotter:
#     """Plots to Visdom"""

#     def __init__(self, env='main'):
#         global vis
#         self.env = env
#         print(f"View visdom results on env {env}")
#         self.plots = {}  # name -> visdom window str
#         self.vis = vis  # not picklable
#         if not self.vis.win_exists('last_updated'):
#             self.vis.win_exists

#     def plot(self, title, series, x, y, setup, xlabel='Epochs', ylabel=None):
#         """
#         title = 'loss' etc
#         series = 'train' etc
#         """
#         hr_min = datetime.datetime.now().strftime("%I:%M")
#         time = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M%p")
#         self.vis.text(
#             f'<b>LAST UPDATED</b><br>{time}', env=self.env, win='last_updated')

#         if setup.expname != 'NoName':
#             title += f" ({setup.expname})"
#         if setup.has_suggestion:
#             title += f" ({setup.sugg_id})"
#         title += f" (Phase {setup.phaser.idx}) "

#         display_title = title
#         if setup.config.sigopt:
#             display_title = f"{display_title}:{setup.sugg_id}"
#         if setup.config.mode is not None:
#             display_title += f" ({setup.config.mode})"
#         display_title += f" ({hr_min})"

#         if title in self.plots:  # update existing plot
#             self.vis.line(
#                 X=np.array([x]),
#                 Y=np.array([y]),
#                 env=self.env,
#                 win=self.plots[title],
#                 name=series,
#                 update='append'
#             )
#         else:  # new plot
#             self.plots[title] = self.vis.line(
#                 X=np.array([x, x]),
#                 Y=np.array([y, y]),
#                 env=self.env,
#                 opts={
#                     'legend': [series],
#                     'title': display_title,
#                     'xlabel': xlabel,
#                     'ylabel': ylabel,
#                 })
#         #mlb.gray("[plotted to visdom]")


# class ProgressBar:
#     def __init__(self, num_steps, num_dots=10):
#         self.num_steps = num_steps
#         self.num_dots = num_dots
#         self.curr_step = 0
#         self.dots_printed = 0

#     def step(self):
#         expected_dots = ceil(self.curr_step/self.num_steps*self.num_dots)
#         dots_to_print = expected_dots - self.dots_printed
#         if dots_to_print > 0:
#             print('.'*dots_to_print, end='', flush=True)
#         self.dots_printed = expected_dots
#         self.curr_step += 1
#         if self.curr_step == self.num_steps:
#             print('!\n', end='', flush=True)


# callback_inputs = []


# def callback(keyword, fn):
#     assert callable(fn)
#     if len(callback_inputs) > 0:
#         if keyword in callback_inputs:
#             callback_inputs.remove(keyword)
#             return fn()
#     while select.select([sys.stdin, ], [], [], 0.0)[0]:
#         try:
#             line = input().strip()
#         except EOFError:
#             return
#         if line.strip() == keyword:
#             return fn()
#         else:
#             callback_inputs.append(line.strip())

# def freezer(keyword='pause'):
#     return callback(keyword, set_trace)
    
# def predicate(keyword):
#     return callback(keyword, lambda:True)

# @contextmanager
# def debug(do_debug=True, ctrlc=(lambda:sys.exit(1)), crash=None):
#     try:
#         yield None
#     except BdbQuit:
#         raise
#     except KeyboardInterrupt:
#         ctrlc()
#         raise
#     except Exception as e:
#         if not do_debug:
#             raise e
#         if crash is not None:
#             red("[CRASH]")
#             crash()
#         print(datetime.datetime.now())
#         print(''.join(tb.format_exception(e.__class__, e, e.__traceback__)))
#         print(format_exception(e, ''))
#         print("curr time:",datetime.datetime.now())
#         post_mortem()
#         sys.exit(1)


# initialize dirs used by espresso
# def init_dirs():
#     dirs = [data_path, error_path]
#     for d in dirs:
#         if not os.path.isdir(d):
#             blue('created:'+d)
#             os.makedirs(d)


# def die(s):
#     raise Exception(mk_red(f"Error:{s}"))


# def warn(s):
#     yellow(f"WARN:{s}")

# # cause nobody likes ugly paths


# def pretty_path(p):
#     return p.replace(homedir, '~')

# # returns ['util','repl','main',...] These are the names of the source modules.
# # this is used in reload_modules()


# def module_ls():
#     files = os.listdir(src_path)
#     files = list(filter(lambda x: x[-3:] == '.py', files))  # only py files
#     mod_names = [x[:-3] for x in files]  # cut off extension
#     return mod_names

# # takes the result of sys.modules as an argument
# # 'verbose' will cause the unformatted exception to be output as well
# # TODO passing in sys.modules may be unnecessary bc util.py may share the same sys.modules
# # as everything else. Worth checking.


# def reload_modules(mods_dict, verbose=False):
#     failed_mods = []
#     for mod in module_ls():
#         if mod in mods_dict:
#             try:
#                 reload(mods_dict[mod])
#                 #blue('reloaded '+mod)
#             except Exception as e:
#                 failed_mods.append(mod)
#                 print(format_exception(e, src_path,
#                                        ignore_outermost=1, verbose=verbose))
#                 pass
#     return failed_mods  # this is TRUTHY if any failed


# class PrettifyErr(Exception):
#     pass
# class VerbatimExc(Exception):
#     pass


# def exception_str(e):
#     return ''.join(tb.format_exception(e.__class__, e, e.__traceback__))

# # Takes a shitty looking normal python exception and beautifies it.
# # also writes to .espresso/error_handling keeping track of the files/line numbers where the errors happened.
# # (which you can automatically open + jump to the correct line using a script like my e() bash function
# # included in a comment near the bottom of this file)
# # relevant_path_piece is just used to remove a lot of irrelevant parts of exceptions.
# # for example it might be '/espresso/src/' to ignore all lines of traceback that dont contain that string
# # "verbose" option will print out the whole original exception in blue followed by
# # the formatted one.
# # ignore_outermost=1 will throw away the first of (fmt,cmd) pair that the program generates, ie the first result of prettify_tb() in the list of results
# # given_text = True if you pass in the actual python exception STRING to as 'e'
# # TODO replace tmpfile with an optional alias list


# def format_exception(e, relevant_path_piece, tmpfile=None, verbose=False, given_text=False, ignore_outermost=0):
#     if verbose:
#         if given_text:
#             blue(''.join(e))
#         else:
#             blue(exception_str(e))
#     try:
#         if given_text:
#             raw_tb = e
#         else:
#             raw_tb = tb.format_exception(e.__class__, e, e.__traceback__)

#         while raw_tb[0] != 'Traceback (most recent call last):\n':
#             # print(raw_tb[0])
#             raw_tb = raw_tb[1:]
#             if raw_tb == []:
#                 return ''
#         raw_tb = raw_tb[1:-1]  # rm first+last. last is a copy of str(e)
#         # blue(''.join(raw_tb))
#         # magenta(raw_tb)
#         formatted = raw_tb

#         # any lines not starting with 'File' are appended to the last seen
#         # line that started with 'File'
#         # (this is for standardization bc the exceptions arent that standardized)
#         # all this should really be done with whatever python under the hood
#         # is generating exceptions haha
#         for (i, s) in enumerate(formatted):
#             if s.strip()[:4] == 'File':
#                 lastfile = i
#             else:
#                 formatted[lastfile] += s
#         # now remove those leftover copies that dont start with File
#         formatted = list(filter(lambda s: s.strip()[:4] == 'File', formatted))

#         # delete everything before the first line that relates to the tmpfile
#         # for (i,s) in enumerate(raw_tb):
#         #    if relevant_path_piece in s:
#         #        formatted = raw_tb[i:]
#         #        break
#         if isinstance(relevant_path_piece, str):
#             formatted = list(
#                 filter(lambda s: relevant_path_piece in s, formatted))
#         elif isinstance(relevant_path_piece, list):
#             def aux(haystack, needles):  # true if haystack contains at least one needle
#                 for n in needles:
#                     if n in haystack:
#                         return True
#                 return False
#             formatted = list(filter(lambda s: aux(
#                 s, relevant_path_piece), formatted))

#         # Turns an ugly traceback segment into a nice one
#         # for a traceback segment that looks like (second line after \n is optional, only shows up sometimes):
#         # File "/Users/matthewbowers/espresso/repl-tmpfiles/113/a_out.py", line 8, in <module>\n      test = 191923j2k9E # syntax error

#         def try_pretty_tb(s):
#             try:
#                 return pretty_tb(s)
#             except Exception as e:
#                 warn("(ignorable) Error during prettifying traceback component.\ncomponent={}\nreason={}".format(
#                     s, exception_str(e)))
#                 return [s, '']

#         def pretty_tb(s):
#             if s[-1] == '\n':
#                 s = s[:-1]
#             if s.count('\n') == 0:
#                 includes_code = includes_arrow = False
#                 msg = s
#             elif s.count('\n') == 1:
#                 includes_code = True
#                 includes_arrow = False
#                 [msg, code_line] = s.split('\n')
#             elif s.count('\n') == 2:
#                 includes_code = includes_arrow = True
#                 [msg, code_line, arrow] = s.split('\n')
#             elif s.count('\n') > 2:
#                 raise PrettifyErr(
#                     'more than 3 lines in a single traceback component:{}')

#             if msg.count(',') == 1:
#                 includes_fn = False
#                 [fpath, lineno] = msg.split(',')
#             elif msg.count(',') == 2:
#                 includes_fn = True
#                 [fpath, lineno, fn_name] = msg.split(',')
#             else:
#                 raise PrettifyErr(
#                     'unexpected number of commas in traceback component line:{}'.format(s))

#             # prettify file name (see example text above function)
#             assert(len(fpath.split('"')) == 3)
#             fpath = fpath.split('"')[1]  # extract actual file name
#             fpath_abs = os.path.abspath(fpath)
#             if fpath == tmpfile:
#                 fname = "tmpfile"
#             else:
#                 fname = os.path.basename(fpath)
#             fpath = fpath.replace(homedir, '~')

#             # prettify line number (see example text above function)
#             lineno = lineno.strip()
#             assert(len(lineno.split(' ')) == 2)
#             lineno_white = lineno.split(' ')[1]
#             lineno = lineno.split(' ')[1]

#             # prettify fn name
#             if includes_fn:
#                 fn_name = fn_name.strip()
#                 assert(len(fn_name.split(' ')) == 2)
#                 fn_name = fn_name.split(' ')[1]
#                 if fn_name == '<fnule>':
#                     fn_name = ''
#                 else:
#                     fn_name = fn_name+'()'
#             else:
#                 fn_name = ''

#             # build final result
#             command = "+{} {}".format(lineno, fpath_abs)
#             result = "{} {} {}".format(
#                 mk_underline(mk_red(fname+" @ "+lineno))+mk_red(':'),
#                 mk_gray('('+fpath+')'),
#                 mk_purple('['+str(try_pretty_tb.cmdcount)+']'),
#                 mk_green(fn_name),
#             )
#             if includes_code:
#                 code_num_spaces = code_line.index(code_line.strip()[0])
#                 code_line = code_line.strip()
#                 lineno_fmtd = '{:>6}: '.format(lineno)
#                 lineno_width = len(lineno_fmtd)
#                 result += '\n{}{}'.format(
#                     mk_bold(mk_green(lineno_fmtd)),
#                     mk_yellow(code_line)
#                 )
#             if includes_arrow:
#                 arrow_num_spaces = arrow.index(arrow.strip()[0])
#                 offset = lineno_width + arrow_num_spaces - code_num_spaces
#                 result += '\n{}{}'.format(' '*offset,
#                                           mk_bold(mk_green('^here')))
#             try_pretty_tb.cmdcount += 1
#             return (result, command)

#         try_pretty_tb.cmdcount = 0
#         res = [try_pretty_tb(s) for s in formatted]
#         res = res[ignore_outermost:]
#         (formatted, commands) = zip(*res)
#         with open(error_path+'/vim_cmds', 'w') as f:
#             commands = list(filter(None, commands))
#             f.write('\n'.join(commands))

#         if given_text:
#             # e[-2][:-1] is the exception str eg 'NameError: name 'foo' is not defined'
#             # silly ['']s are just to add extra newlines
#             formatted = [''] + [mk_red(e[-2][:-1])] + list(formatted) + ['']
#         else:
#             formatted = [''] + [mk_red(e)] + list(formatted) + ['']

#         return '\n'.join(formatted)
#     except Exception as e2:
#         warn("(ignorable) Failed to Prettify exception, using default format. Note that the prettifying failure was due to: {}".format(
#             exception_str(e2)))
#         print(e)
#         if given_text:
#             return (mk_red(e), [])
#         else:
#             return (mk_red(exception_str(e)), [])




# from colorama import Fore, Back, Style, init

# init()
# # Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# # Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# # Style: DIM, NORMAL, BRIGHT, RESET_ALL.

# # \001 and \002 are used by readline to fix buggy prompt displaying when
# # browsing history (https://stackoverflow.com/questions/9468435/look-how-to-fix-column-calculation-in-python-readline-if-use-color-prompt)


# def color(s, c):
#     return '\001'+c+'\002'+str(s)+'\001'+Style.RESET_ALL+'\002'

# colors = {
#     'green': Fore.GREEN,
#     'red': Fore.RED,
#     'purple': Fore.MAGENTA,
#     'cyan': Fore.CYAN,
#     'blue': Fore.BLUE,
#     'yellow': Fore.YELLOW,
#     'gray': '\033[90m',
#     'bold': '\033[1m',
#     'underline': '\033[4m',
#     'none': Style.RESET_ALL
# }

# # color a string

# def mk_green(s):
#     return color(s, Fore.GREEN)

# def mk_red(s):
#     return color(s, Fore.RED)

# def mk_purple(s):
#     return color(s, Fore.MAGENTA)
# def mk_magenta(s):
#     return mk_purple(s)

# def mk_blue(s):
#     return color(s, Fore.BLUE)

# def mk_cyan(s):
#     return color(s, Fore.CYAN)

# def mk_yellow(s):
#     return color(s, Fore.YELLOW)

# def mk_gray(s):
#     return color(s, '\033[90m')

# def mk_bold(s):
#     return color(s, '\033[1m')

# def mk_underline(s):
#     return color(s, '\033[4m')

# # color a string then print it immediately


# def green(s):
#     print(mk_green(s))

# def red(s):
#     print(mk_red(s))

# def purple(s):
#     print(mk_purple(s))
# def magenta(s):
#     purple(s)

# def blue(s):
#     print(mk_blue(s))

# def cyan(s):
#     print(mk_cyan(s))

# def yellow(s):
#     print(mk_yellow(s))

# def gray(s):
#     print(mk_gray(s))

# # add style to a string the print it immediately


# def bold(s):
#     print(mk_bold(s))


# def underline(s):
#     print(mk_underline(s))

# # color and BOLD a string then print it immediately

# def bgreen(s):
#     print(mk_green(mk_bold(s)))

# def bred(s):
#     print(mk_red(mk_bold(s)))

# def bpurple(s):
#     print(mk_purple(mk_bold(s)))

# def bblue(s):
#     print(mk_blue(mk_bold(s)))

# def bcyan(s):
#     print(mk_cyan(mk_bold(s)))

# def byellow(s):
#     print(mk_yellow(mk_bold(s)))

# def bgray(s):
#     print(mk_gray(mk_bold(s)))


# you can actually run this file and use it to format any exception you get.
# I have it running on my main python by having the following in my
# ~/.bash_profile
# p3(){
#    if [ $# -eq 0 ]; then
#        python3
#    else
#        python3 $* 2> ~/.py_err
#        python3 ~/espresso/src/util.py $1 ~/.py_err
#    fi
# }
# v3(){
#    echo "verbose" > ~/.py_err
#    python3 $* 2>> ~/.py_err
#    python3 ~/espresso/src/util.py $1 ~/.py_err
# }
# opens file based on python error idx passed to it
# defaults to last error
# e(){
#    if [ $# -eq 0 ]; then
#        line=$(tail -n1 ~/.espresso/error_handling/vim_cmds)
#    else
#        num=$(( $1 + 1  ))
#        line=$(sed -n -e "$num"p ~/.espresso/error_handling/vim_cmds)
#    fi
#    nvim $line
# }

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 3:
#         red("call like: p3 "+sys.argv[0] +
#             " /path/to/file.py /path/to/stderr/file")
#     if len(sys.argv) == 5:  # optionally pass a relevant path piece. otherwise it'll assume no relevant piece (often this is ok bc exceptions dont always contain full paths anyways
#         relevant_path_piece = sys.argv[3]
#     file = sys.argv[1]
#     #relevant_path_piece = os.environ['HOME']
#     relevant_path_piece = ''
#     stderr = sys.argv[2]
#     with open(stderr, 'r') as f:
#         exception = f.read().split('\n')
#     if exception == ['']:
#         exit(0)
#     verbose = False
#     if exception[0] == 'verbose':
#         verbose = True
#         exception = exception[1:]
#     exception = [line+'\n' for line in exception]
#     fmtd = format_exception(exception, relevant_path_piece,
#                             given_text=True, verbose=verbose)
#     print(fmtd)


# init_dirs()
