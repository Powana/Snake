import numpy as np
import random
width = 5
height = 5
z = np.array([[1, 2, 3, 0, 11], [4, 5, 6, 2, 12], [7, 8, 9, 1, 13], [2, 3, 4, 5, 14], [9, 10, 11, 12, 13]])
z[0][4] = 3
z[1][3] = 2
z[2][2] = 1
print(z)
x = 2
max_x = width - x - 1  # 2
y = 3
max_y = height - y - 1  # 4

h_c = [y, x]

c_x = x + 1
c_y = height - y

print("Conventional", c_x, c_y)

# Northwest
r = -1
if (c_x + c_y) == (width + 1):
    r = c_x - 1
elif (c_x + c_y) < (width + 1):
    r = c_x - 1
elif (c_x + c_y) > (width + 1):
    r = width - c_y

print("Start:", h_c, z[h_c[0]][h_c[1]])
print(r)


print("==========")

for n in range(r):

    print(h_c[1]-n-1, h_c[0]-n-1, z[h_c[0] - n - 1][h_c[1] - n - 1])

# for i in range(5-2-1):
#     print(z[h_c[0]+i+1][h_c[1]+i+1])