import matplotlib.pyplot as plt
#
# 6.19
# 5.20
#
# 19.16
# 12.86
#
# 32.48
# 20.42
#
# 44.72
# 27.70

# x = range(4)
# y1 = [6.19, 19.16, 32.48, 44.72]
# y2 = [5.20, 12.86, 20.42, 27.70]
# fig = plt.figure()
# # ax1 = fig.add_subplot(111)
# #
# # ax1.scatter(x, y1, s=15, c='b', marker="s", label='Change-data')
# # ax1.scatter(x, y2, s=15, c='r', marker="o", label='PR-data')
# # # ax1.lineplot(x, y1, s=5, c='b')
# # # ax1.line(x, y2, s=5, c='r')
# # plt.legend(loc='upper left');
# plt.plot(x, y1, 'b*-', label='Code-Change-Data')
# plt.plot(x, y2, 'ro-', label='PR-Data')
# plt.legend(loc='best')
# plt.xlabel('Edit distance from correct patch')
# plt.ylabel('Cumulative percentage')
# plt.xticks(x)
# plt.grid(b=True, which='both', linewidth=0.5)
# plt.savefig(fname='newar-miss.pdf')
# plt.show()
x = [5,10,15,20,25,30,35,40,45,50,55,60]
y = [0,18,33,48,48,48,48,48,48,48,51,51]
plt.figure()
plt.plot(x, y, 'b*-')
# plt.legend(loc='best')
plt.xlabel('Time (Minutes)', fontsize=16)
plt.ylabel('Number of Patches', fontsize=16)
plt.xticks(x,fontsize=12)
plt.yticks(fontsize=12)
plt.grid(b=True, which='both', linewidth=1)
plt.savefig(fname='defj-test-patch.pdf')
plt.show()
