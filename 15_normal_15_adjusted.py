import numpy as np
import matplotlib.pyplot as plt

x = np.array([])
y = np.array([])

for mu in range(60,140):
    sigma = 20
    num = 50000

    rand_data = np.random.normal(mu, sigma, num)

    res1 = 0
    res2 = 0

    for r in rand_data:
        if np.random.rand(1) < 1 / 2:
            if r < 80:
                res += 2 ** (-5 * np.random.rand(1))
        else:
            if r < 140:
                res += 2 ** (-5 * np.random.rand(1))

    x = np.append(x, mu)
    y = np.append(y, res1 + 0.6 * res2 - 4750)

plt.title("revenue-distance graph using 15 small fish vessels and 15 adjusted") 
plt.xlabel("the best temperature's distance to coastline (km)") 
plt.ylabel("the revenue per day (fish)") 
plt.plot(x,y)
plt.show()