import subprocess as sp
import os

"""
Set up self mailing by adding ~/.mlb.email and ~/.mlb.phone to have contents like 'myemail@foo.bar' and '72837849279@vtext.com'
"""



def text_me(body,verbose=False):
    return email(body, subject='',to=get_phone(),verbose=verbose)
def email_me(subject,body,verbose=False):
    return email(body, subject=subject,to=get_email(),verbose=verbose)


def email(body,subject,to,verbose=False):
    msg = '''\
To: {to}
Subject: {subject}

{body}\
'''.format(to=to,subject=subject,body=body).encode('utf-8')
    if verbose: print(msg)

    proc = sp.Popen(['sendmail','-t'],stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE)
    try:
        out, err = proc.communicate(input=msg,timeout=10)
    except:
        proc.kill()
        out, err = proc.communicate()
    if verbose:
        print('[sendmail stdout:] ', out)
        print('[sendmail stderr:] ', err)
    return out, err

def get_email():
    try:
        with open(os.environ['HOME']+'/.mlb.email') as f:
            return f.read().strip()
    except FileNotFoundError as e:
        raise Exception(f"{e} | Set up self mailing by adding ~/.mlb.email and ~/.mlb.phone to have contents like 'myemail@foo.bar' and '72837849279@vtext.com'")

def get_phone():
    try:
        with open(os.environ['HOME']+'/.mlb.phone') as f:
            return f.read().strip()
    except FileNotFoundError as e:
        raise Exception(f"{e} Set up self mailing by adding ~/.mlb.email and ~/.mlb.phone to have contents like 'myemail@foo.bar' and '72837849279@vtext.com'")
