import os
import time
import subprocess
import datetime
import signal
import argparse
import locale

encoding = locale.getdefaultlocale()[1]
def process(file_path, start, end):
    i = start
    timeout = 5
    files = os.listdir(file_path)
    print(len(files))
    if end > len(files):
        end = len(files)
    while i < end:
        slicer = "bash ./joern/slicer.sh " + file_path + "  " + str(files[i]) + "  1 " + "parsed/" + str(files[i])
        start0 = datetime.datetime.now()
        process1 = subprocess.Popen(slicer, shell = True,encoding=encoding)
        while process1.poll() is None:
            time.sleep(0.2)
            end0 = datetime.datetime.now()
            if (end0-start0).seconds > timeout:
                os.kill(process1.pid, signal.SIGKILL)
                os.waitpid(-1, os.WNOHANG)
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='funtions dic.', default='./devign_dataset/devign_raw_code')
    parser.add_argument('--start', help='start functions number to parsed', type=int, default=0)
    parser.add_argument('--end', help='end functions number to parsed', type=int, default=1)
    args = parser.parse_args()
    file_path = args.file_path
    start = args.start
    end = args.end
    process(file_path, start, end)
