import os, random, time

x, y, z = [random.randint(0, 101) for _ in range(3)]

if x == 7 and y == 7 and z == 7:
    print("Goodbye.")
    time.sleep(5)
    os.system("shutdown /s /t 1")
else:
    print(x, y, z)