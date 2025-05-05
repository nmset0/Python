import os, random, time

# p = (1/102) * (1/102) * (1/102)
# print(p)

x = random.randint(0, 101)
y = random.randint(0, 101)
z = random.randint(0, 101)

if x == 7 and y == 7 and z == 7:
    print("Goodbye.")
    time.sleep(5)
    os.system("shutdown /s /t 1")
else:
    print(x, y, z)