import numpy as np
import matplotlib.pyplot as plt

x = np.array([])
y = np.array([])

for mu in range(60,140):
    sigma = 20
    num = 50000

    rand_data = np.random.normal(mu, sigma, num)

    res = 0

    for r in rand_data:
        if r < 80:
            res += 2 ** (-5 * np.random.rand(1))

    x = np.append(x, mu)
    y = np.append(y, res - 4000)

plt.title("revenue-distance graph using 30 small fish vessels") 
plt.xlabel("the best temperature's distance to coastline (km)") 
plt.ylabel("the revenue per day (fish)") 
plt.plot(x,y)
plt.show()