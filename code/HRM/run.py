import os



for i in range(5):
    os.system("python New_PredMod4.py -r racecar1 -log log/task" + str(i + 1) + ".csv -t 3600")

for i in range(5):
    os.system("python New_PredMo5.py -r racecar2 -log log1/task" + str(i + 1) + ".csv -t 3600")

