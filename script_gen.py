import random

syl_dict = {1: "AA", 2: "OO", 3: "FF", 4: "MM"}

script = ""

for i in range(100):
    num = random.randint(1, 4)
    script += (syl_dict[num] + " ")

print(script)