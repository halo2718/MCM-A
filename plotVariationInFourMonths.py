list_t= get_temperature(0, -3)

y = []
for i in range(1800):
    if i%12 == 6:
        y.append(list_t[i])

yb = []
for i in range(1800):
    if i%12 == 3:
        yb.append(list_t[i])
        
yc = []
for i in range(1800):
    if i%12 == 9:
        yc.append(list_t[i])

ya = []
for i in range(1800):
    if i%12 == 0:
        ya.append(list_t[i])

x = np.arange(1870,2020)
fig, ax = plt.subplots(figsize = (4,3))
plt.plot(x,y)
plt.plot(x,ya)
plt.plot(x,yb)
plt.plot(x,yc)
ax.set_ylabel('Temperature', fontsize = 9)
ax.set_xlabel('Year', fontsize = 9) 
plt.savefig('ori.eps', bbox_inches='tight')