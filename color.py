
from colorama import Fore, Back, Style, init

init()
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL.

# \001 and \002 are used by readline to fix buggy prompt displaying when
# browsing history (https://stackoverflow.com/questions/9468435/look-how-to-fix-column-calculation-in-python-readline-if-use-color-prompt)


def color(s, c):
    return '\001'+c+'\002'+str(s)+'\001'+Style.RESET_ALL+'\002'

colors = {
    'green': Fore.GREEN,
    'red': Fore.RED,
    'purple': Fore.MAGENTA,
    'cyan': Fore.CYAN,
    'blue': Fore.BLUE,
    'yellow': Fore.YELLOW,
    'gray': '\033[90m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    'none': Style.RESET_ALL
}

# color a string

def mk_green(s):
    return color(s, Fore.GREEN)

def mk_red(s):
    return color(s, Fore.RED)

def mk_purple(s):
    return color(s, Fore.MAGENTA)
def mk_magenta(s):
    return mk_purple(s)

def mk_blue(s):
    return color(s, Fore.BLUE)

def mk_cyan(s):
    return color(s, Fore.CYAN)

def mk_yellow(s):
    return color(s, Fore.YELLOW)

def mk_gray(s):
    return color(s, '\033[90m')

def mk_bold(s):
    return color(s, '\033[1m')

def mk_underline(s):
    return color(s, '\033[4m')

# color a string then print it immediately


def green(s):
    print(mk_green(s))

def red(s):
    print(mk_red(s))

def purple(s):
    print(mk_purple(s))
def magenta(s):
    purple(s)

def blue(s):
    print(mk_blue(s))

def cyan(s):
    print(mk_cyan(s))

def yellow(s):
    print(mk_yellow(s))

def gray(s):
    print(mk_gray(s))

# add style to a string the print it immediately


def bold(s):
    print(mk_bold(s))


def underline(s):
    print(mk_underline(s))

# color and BOLD a string then print it immediately

def bgreen(s):
    print(mk_green(mk_bold(s)))

def bred(s):
    print(mk_red(mk_bold(s)))

def bpurple(s):
    print(mk_purple(mk_bold(s)))

def bblue(s):
    print(mk_blue(mk_bold(s)))

def bcyan(s):
    print(mk_cyan(mk_bold(s)))

def byellow(s):
    print(mk_yellow(mk_bold(s)))

def bgray(s):
    print(mk_gray(mk_bold(s)))

