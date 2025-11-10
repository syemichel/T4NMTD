import os
import time

for i in range(10):
    os.system("python T4DMT_PPO.py -r racecar1 -log log/task" + str(i + 1) + "/task" + str(i + 1) + ".csv" + " -t 5400 -o 5")
