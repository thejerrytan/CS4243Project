import numpy as np

for k in range(0, 5):
    points = []
    noise_interval = 50

    jump_count = 0

    for i in range(-500, 500):
        noise = 0
        if i % (noise_interval + 0) == 0:
            jump_count += 1
            noise = 5
        if i % (noise_interval + 1) == 0:
            noise = 10
        if i % (noise_interval + 2) == 0:
            noise = 15
        if i % (noise_interval + 3) == 0:
            noise = 20
        if i % (noise_interval + 4) == 0:
            noise = 15
        if i % (noise_interval + 5) == 0:
            noise = 10
        if i % (noise_interval + 6) == 0:
            noise = 3

        # print k
        points.append([i + (k * 50) + noise, i + (k - 50) + noise])

    print "jump count: ", jump_count
    np.savetxt("./plot/clip1_player_%d.txt" % k, points, fmt='%1.4f')
