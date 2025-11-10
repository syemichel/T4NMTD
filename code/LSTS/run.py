import os
import time

import os


time.sleep(20000)
for i in range(5):
    os.system("python LSTS4.py -r racecar1 -i inst1 -log log/task" + str(i + 1) + ".csv" + " -t 3600")

for i in range(5):
    os.system("python LSTS5.py -r racecar2 -i inst1 -log log1/task" + str(i + 1) + ".csv" + " -t 3600")