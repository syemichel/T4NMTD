import os
import time


for i in range(4):
    os.system("python HDQN5.py -r racecar2 -i inst1 -log log1/task" + str(i + 2) + ".csv" + " -t 3600")