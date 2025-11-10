import os
import time

for i in range(5):
    os.system("python MOD_SAC.py -r racecar1 -log log/task" + str(i + 1) + "/task" + str(i + 1) + ".csv" + " -t 3600 -c 4")

for i in range(5):
    os.system("python MOD_SAC.py -r racecar2 -log log1/task" + str(i + 1) + "/task" + str(i + 1) + ".csv" + " -t 5400 -c 16")