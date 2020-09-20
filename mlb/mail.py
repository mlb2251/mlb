import subprocess as sp

# Send yourself an email!. Examples:
# email_me('heres an email!')
# text_me('heres a text')
# email_me('im overriding the reciever field now',to=somebodyelse@columbia.edu)
# email_me('overriding the subject line',subject='new subject line')
# Note that verbose=False can be used to prevent stderr and stdout from being
# shown

def text_me(body,subject='',to='7743120116@vtext.com',verbose=True):
    return email_me(body,subject=subject,to=to,verbose=verbose)

def email_me(body,subject='[py]',to='mlb2251@columbia.edu',verbose=True):
    msg = '''\
To: {to}
Subject: {subject}

{body}\
'''.format(to=to,subject=subject,body=body).encode('utf-8')
    print(msg)

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

