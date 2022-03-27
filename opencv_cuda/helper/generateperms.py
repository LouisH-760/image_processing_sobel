xsob = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

ysob = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]]

print("// auto generayed for x")

for i in range(-1, 2):
    for j in range(-1, 2):
        print(
            f"x += {xsob[i + 1][j + 1]} * img[({i} + currRow) * cols + ({j} + currCol)];")

print("// auto generayed for y")

for i in range(-1, 2):
    for j in range(-1, 2):
        print(
            f"y += {ysob[i + 1][j + 1]} * img[({i} + currRow) * cols + ({j} + currCol)];")
