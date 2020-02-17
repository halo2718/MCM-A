list_t= get_temperature(0, -3)
y = []
for i in range(1800):
    if i%12 == 6:
        y.append(list_t[i])

x = np.arange(1870,2020,5)
print(len(x))

%config InlineBackend.figure_format = 'svg'
fig, ax = plt.subplots(figsize = (4,3))
plt.plot(x,result)
ax.set_ylabel('Temperature', fontsize = 9)
ax.set_xlabel('Year', fontsize = 9) 
plt.savefig('5con.eps', bbox_inches='tight')