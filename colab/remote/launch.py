"""Runs ON the Colab VM through `colab exec`: start a shell script detached, so it survives the exec call.

Usage inside an exec file: sys.argv = ['launch', '/content/jobs.sh', '/content/jobs.log', *args]; exec(...)
"""
import os
import subprocess
import sys

script, log, *rest = sys.argv[1:]
process = subprocess.Popen(['bash', script, *rest], stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                           start_new_session=True, env=dict(os.environ), cwd='/content')
print('launched', process.pid)
