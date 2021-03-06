
import sys,os
from mlb.color import *
from mlb.core import *

import datetime
from pdb import post_mortem
from pdb import set_trace
import ast
from itertools import zip_longest
from importlib import reload

import numpy as np
import inspect
import subprocess as sp
import os
import sys
import select
from bdb import BdbQuit
from contextlib import contextmanager
from pdb import post_mortem, set_trace
import traceback as tb
import time
from math import ceil


homedir = os.environ['HOME']
src_path = os.path.dirname(os.path.realpath(__file__))+'/'
#src_path = homedir+'/espresso/src/'
data_path = os.path.join(homedir, '.espresso/')
error_path = os.path.join(data_path+'error_handling/')
#repl_path = data_path+'repl-tmpfiles/'
#pipe_dir = data_path+'pipes/'
pwd_file = os.path.join(data_path, '.{}.PWD'.format(os.getpid()))


def init_dirs():
    dirs = [data_path, error_path]
    for d in dirs:
        if not os.path.isdir(d):
            blue('created:'+d)
            os.makedirs(d)


class PrettifyErr(Exception):
    pass
class VerbatimExc(Exception):
    pass

def exception_str(e):
    return ''.join(tb.format_exception(e.__class__, e, e.__traceback__))

# Takes a shitty looking normal python exception and beautifies it.
# also writes to .espresso/error_handling keeping track of the files/line numbers where the errors happened.
# (which you can automatically open + jump to the correct line using a script like my e() bash function
# included in a comment near the bottom of this file)
# relevant_path_piece is just used to remove a lot of irrelevant parts of exceptions.
# for example it might be '/espresso/src/' to ignore all lines of traceback that dont contain that string
# "verbose" option will print out the whole original exception in blue followed by
# the formatted one.
# ignore_outermost=1 will throw away the first of (fmt,cmd) pair that the program generates, ie the first result of prettify_tb() in the list of results
# given_text = True if you pass in the actual python exception STRING to as 'e'
# TODO replace tmpfile with an optional alias list


def format_exception(e, relevant_path_piece, tmpfile=None, verbose=False, given_text=False, ignore_outermost=0):
    if verbose:
        if given_text:
            blue(''.join(e))
        else:
            blue(exception_str(e))
    try:
        if given_text:
            raw_tb = e
        else:
            raw_tb = tb.format_exception(e.__class__, e, e.__traceback__)

        while raw_tb[0] != 'Traceback (most recent call last):\n':
            # print(raw_tb[0])
            raw_tb = raw_tb[1:]
            if raw_tb == []:
                return ''
        raw_tb = raw_tb[1:-1]  # rm first+last. last is a copy of str(e)
        # blue(''.join(raw_tb))
        # magenta(raw_tb)
        formatted = raw_tb

        # any lines not starting with 'File' are appended to the last seen
        # line that started with 'File'
        # (this is for standardization bc the exceptions arent that standardized)
        # all this should really be done with whatever python under the hood
        # is generating exceptions haha
        for (i, s) in enumerate(formatted):
            if s.strip()[:4] == 'File':
                lastfile = i
            else:
                formatted[lastfile] += s
        # now remove those leftover copies that dont start with File
        formatted = list(filter(lambda s: s.strip()[:4] == 'File', formatted))

        # delete everything before the first line that relates to the tmpfile
        # for (i,s) in enumerate(raw_tb):
        #    if relevant_path_piece in s:
        #        formatted = raw_tb[i:]
        #        break
        if isinstance(relevant_path_piece, str):
            formatted = list(
                filter(lambda s: relevant_path_piece in s, formatted))
        elif isinstance(relevant_path_piece, list):
            def aux(haystack, needles):  # true if haystack contains at least one needle
                for n in needles:
                    if n in haystack:
                        return True
                return False
            formatted = list(filter(lambda s: aux(
                s, relevant_path_piece), formatted))

        # Turns an ugly traceback segment into a nice one
        # for a traceback segment that looks like (second line after \n is optional, only shows up sometimes):
        # File "/Users/matthewbowers/espresso/repl-tmpfiles/113/a_out.py", line 8, in <module>\n      test = 191923j2k9E # syntax error

        def try_pretty_tb(s):
            try:
                return pretty_tb(s)
            except Exception as e:
                warn("(ignorable) Error during prettifying traceback component.\ncomponent={}\nreason={}".format(
                    s, exception_str(e)))
                return [s, '']

        def pretty_tb(s):
            if s[-1] == '\n':
                s = s[:-1]
            if s.count('\n') == 0:
                includes_code = includes_arrow = False
                msg = s
            elif s.count('\n') == 1:
                includes_code = True
                includes_arrow = False
                [msg, code_line] = s.split('\n')
            elif s.count('\n') == 2:
                includes_code = includes_arrow = True
                [msg, code_line, arrow] = s.split('\n')
            elif s.count('\n') > 2:
                raise PrettifyErr(
                    'more than 3 lines in a single traceback component:{}')

            if msg.count(',') == 1:
                includes_fn = False
                [fpath, lineno] = msg.split(',')
            elif msg.count(',') == 2:
                includes_fn = True
                [fpath, lineno, fn_name] = msg.split(',')
            else:
                raise PrettifyErr(
                    'unexpected number of commas in traceback component line:{}'.format(s))

            # prettify file name (see example text above function)
            assert(len(fpath.split('"')) == 3)
            fpath = fpath.split('"')[1]  # extract actual file name
            fpath_abs = os.path.abspath(fpath)
            if fpath == tmpfile:
                fname = "tmpfile"
            else:
                fname = os.path.basename(fpath)
            fpath = fpath.replace(homedir, '~')

            # prettify line number (see example text above function)
            lineno = lineno.strip()
            assert(len(lineno.split(' ')) == 2)
            lineno_white = lineno.split(' ')[1]
            lineno = lineno.split(' ')[1]

            # prettify fn name
            if includes_fn:
                fn_name = fn_name.strip()
                assert(len(fn_name.split(' ')) == 2)
                fn_name = fn_name.split(' ')[1]
                if fn_name == '<fnule>':
                    fn_name = ''
                else:
                    fn_name = fn_name+'()'
            else:
                fn_name = ''

            # build final result
            command = "+{} {}".format(lineno, fpath_abs)
            result = "{} {} {}".format(
                mk_underline(mk_red(fname+" @ "+lineno))+mk_red(':'),
                mk_gray('('+fpath+')'),
                mk_purple('['+str(try_pretty_tb.cmdcount)+']'),
                mk_green(fn_name),
            )
            if includes_code:
                code_num_spaces = code_line.index(code_line.strip()[0])
                code_line = code_line.strip()
                lineno_fmtd = '{:>6}: '.format(lineno)
                lineno_width = len(lineno_fmtd)
                result += '\n{}{}'.format(
                    mk_bold(mk_green(lineno_fmtd)),
                    mk_yellow(code_line)
                )
            if includes_arrow:
                arrow_num_spaces = arrow.index(arrow.strip()[0])
                offset = lineno_width + arrow_num_spaces - code_num_spaces
                result += '\n{}{}'.format(' '*offset,
                                          mk_bold(mk_green('^here')))
            try_pretty_tb.cmdcount += 1
            return (result, command)

        try_pretty_tb.cmdcount = 0
        res = [try_pretty_tb(s) for s in formatted]
        res = res[ignore_outermost:]
        (formatted, commands) = zip(*res)
        with open(error_path+'/vim_cmds', 'w') as f:
            commands = list(filter(None, commands))
            f.write('\n'.join(commands))

        if given_text:
            # e[-2][:-1] is the exception str eg 'NameError: name 'foo' is not defined'
            # silly ['']s are just to add extra newlines
            formatted = [''] + [mk_red(e[-2][:-1])] + list(formatted) + ['']
        else:
            formatted = [''] + [mk_red(e)] + list(formatted) + ['']

        return '\n'.join(formatted)
    except Exception as e2:
        warn("(ignorable) Failed to Prettify exception, using default format. Note that the prettifying failure was due to: {}".format(
            exception_str(e2)))
        print(e)
        if given_text:
            return (mk_red(e), [])
        else:
            return (mk_red(exception_str(e)), [])

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


class PrettifyErr(Exception):
    pass
class VerbatimExc(Exception):
    pass


def exception_str(e):
    return ''.join(tb.format_exception(e.__class__, e, e.__traceback__))

# Takes a shitty looking normal python exception and beautifies it.
# also writes to .espresso/error_handling keeping track of the files/line numbers where the errors happened.
# (which you can automatically open + jump to the correct line using a script like my e() bash function
# included in a comment near the bottom of this file)
# relevant_path_piece is just used to remove a lot of irrelevant parts of exceptions.
# for example it might be '/espresso/src/' to ignore all lines of traceback that dont contain that string
# "verbose" option will print out the whole original exception in blue followed by
# the formatted one.
# ignore_outermost=1 will throw away the first of (fmt,cmd) pair that the program generates, ie the first result of prettify_tb() in the list of results
# given_text = True if you pass in the actual python exception STRING to as 'e'
# TODO replace tmpfile with an optional alias list


def format_exception(e, relevant_path_piece, tmpfile=None, verbose=False, given_text=False, ignore_outermost=0):
    if verbose:
        if given_text:
            blue(''.join(e))
        else:
            blue(exception_str(e))
    try:
        if given_text:
            raw_tb = e
        else:
            raw_tb = tb.format_exception(e.__class__, e, e.__traceback__)

        while raw_tb[0] != 'Traceback (most recent call last):\n':
            # print(raw_tb[0])
            raw_tb = raw_tb[1:]
            if raw_tb == []:
                return ''
        raw_tb = raw_tb[1:-1]  # rm first+last. last is a copy of str(e)
        # blue(''.join(raw_tb))
        # magenta(raw_tb)
        formatted = raw_tb

        # any lines not starting with 'File' are appended to the last seen
        # line that started with 'File'
        # (this is for standardization bc the exceptions arent that standardized)
        # all this should really be done with whatever python under the hood
        # is generating exceptions haha
        for (i, s) in enumerate(formatted):
            if s.strip()[:4] == 'File':
                lastfile = i
            else:
                formatted[lastfile] += s
        # now remove those leftover copies that dont start with File
        formatted = list(filter(lambda s: s.strip()[:4] == 'File', formatted))

        # delete everything before the first line that relates to the tmpfile
        # for (i,s) in enumerate(raw_tb):
        #    if relevant_path_piece in s:
        #        formatted = raw_tb[i:]
        #        break
        if isinstance(relevant_path_piece, str):
            formatted = list(
                filter(lambda s: relevant_path_piece in s, formatted))
        elif isinstance(relevant_path_piece, list):
            def aux(haystack, needles):  # true if haystack contains at least one needle
                for n in needles:
                    if n in haystack:
                        return True
                return False
            formatted = list(filter(lambda s: aux(
                s, relevant_path_piece), formatted))

        # Turns an ugly traceback segment into a nice one
        # for a traceback segment that looks like (second line after \n is optional, only shows up sometimes):
        # File "/Users/matthewbowers/espresso/repl-tmpfiles/113/a_out.py", line 8, in <module>\n      test = 191923j2k9E # syntax error

        def try_pretty_tb(s):
            try:
                return pretty_tb(s)
            except Exception as e:
                warn("(ignorable) Error during prettifying traceback component.\ncomponent={}\nreason={}".format(
                    s, exception_str(e)))
                return [s, '']

        def pretty_tb(s):
            if s[-1] == '\n':
                s = s[:-1]
            if s.count('\n') == 0:
                includes_code = includes_arrow = False
                msg = s
            elif s.count('\n') == 1:
                includes_code = True
                includes_arrow = False
                [msg, code_line] = s.split('\n')
            elif s.count('\n') == 2:
                includes_code = includes_arrow = True
                [msg, code_line, arrow] = s.split('\n')
            elif s.count('\n') > 2:
                raise PrettifyErr(
                    'more than 3 lines in a single traceback component:{}')

            if msg.count(',') == 1:
                includes_fn = False
                [fpath, lineno] = msg.split(',')
            elif msg.count(',') == 2:
                includes_fn = True
                [fpath, lineno, fn_name] = msg.split(',')
            else:
                raise PrettifyErr(
                    'unexpected number of commas in traceback component line:{}'.format(s))

            # prettify file name (see example text above function)
            assert(len(fpath.split('"')) == 3)
            fpath = fpath.split('"')[1]  # extract actual file name
            fpath_abs = os.path.abspath(fpath)
            if fpath == tmpfile:
                fname = "tmpfile"
            else:
                fname = os.path.basename(fpath)
            fpath = fpath.replace(homedir, '~')

            # prettify line number (see example text above function)
            lineno = lineno.strip()
            assert(len(lineno.split(' ')) == 2)
            lineno_white = lineno.split(' ')[1]
            lineno = lineno.split(' ')[1]

            # prettify fn name
            if includes_fn:
                fn_name = fn_name.strip()
                assert(len(fn_name.split(' ')) == 2)
                fn_name = fn_name.split(' ')[1]
                if fn_name == '<fnule>':
                    fn_name = ''
                else:
                    fn_name = fn_name+'()'
            else:
                fn_name = ''

            # build final result
            command = "+{} {}".format(lineno, fpath_abs)
            result = "{} {} {}".format(
                mk_underline(mk_red(fname+" @ "+lineno))+mk_red(':'),
                mk_gray('('+fpath+')'),
                mk_purple('['+str(try_pretty_tb.cmdcount)+']'),
                mk_green(fn_name),
            )
            if includes_code:
                code_num_spaces = code_line.index(code_line.strip()[0])
                code_line = code_line.strip()
                lineno_fmtd = '{:>6}: '.format(lineno)
                lineno_width = len(lineno_fmtd)
                result += '\n{}{}'.format(
                    mk_bold(mk_green(lineno_fmtd)),
                    mk_yellow(code_line)
                )
            if includes_arrow:
                arrow_num_spaces = arrow.index(arrow.strip()[0])
                offset = lineno_width + arrow_num_spaces - code_num_spaces
                result += '\n{}{}'.format(' '*offset,
                                          mk_bold(mk_green('^here')))
            try_pretty_tb.cmdcount += 1
            return (result, command)

        try_pretty_tb.cmdcount = 0
        res = [try_pretty_tb(s) for s in formatted]
        res = res[ignore_outermost:]
        (formatted, commands) = zip(*res)
        with open(error_path+'/vim_cmds', 'w') as f:
            commands = list(filter(None, commands))
            f.write('\n'.join(commands))

        if given_text:
            # e[-2][:-1] is the exception str eg 'NameError: name 'foo' is not defined'
            # silly ['']s are just to add extra newlines
            formatted = [''] + [mk_red(e[-2][:-1])] + list(formatted) + ['']
        else:
            formatted = [''] + [mk_red(e)] + list(formatted) + ['']

        return '\n'.join(formatted)
    except Exception as e2:
        warn("(ignorable) Failed to Prettify exception, using default format. Note that the prettifying failure was due to: {}".format(
            exception_str(e2)))
        print(e)
        if given_text:
            return (mk_red(e), [])
        else:
            return (mk_red(exception_str(e)), [])


init_dirs()
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        red("call like: p3 "+sys.argv[0] +
            " /path/to/file.py /path/to/stderr/file")
    if len(sys.argv) == 5:  # optionally pass a relevant path piece. otherwise it'll assume no relevant piece (often this is ok bc exceptions dont always contain full paths anyways
        relevant_path_piece = sys.argv[3]
    file = sys.argv[1]
    #relevant_path_piece = os.environ['HOME']
    relevant_path_piece = ''
    stderr = sys.argv[2]
    with open(stderr, 'r') as f:
        exception = f.read().split('\n')
    if exception == ['']:
        exit(0)
    verbose = False
    if exception[0] == 'verbose':
        verbose = True
        exception = exception[1:]
    exception = [line+'\n' for line in exception]
    fmtd = format_exception(exception, relevant_path_piece,
                            given_text=True, verbose=verbose)
    print(fmtd)


