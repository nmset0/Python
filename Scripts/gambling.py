import os, random, time


num = random.randint(0, 101)

if num == 0:
    print("Goodbye.")
    time.sleep(5)
    os.system("shutdown /s /t 1")
else:
    print(num)