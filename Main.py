from time import time
import SVC
import Neural_Network
import matplotlib.pyplot as plt
#
#
# # ================First Approach: clustering using SVC================
# SVC.main()
#
# # ========================Second Approach: NN========================
# Neural_Network.main()
#
# # ===========Third Approach: dimension reduction using PCA===========
# # SVC
# acc = []
# timelist = []
# dim = list(range(3, 21))
# for i in dim:
#     t1 = time()
#     acc.append(SVC.main(dr=i))
#     t2 = time()
#     timelist.append(t2-t1)
#     print(t2-t1)
#
# t1 = time()
# acc.append(SVC.main(dr=0))
# t2 = time()
# timelist.append(t2-t1)
# print(t2-t1)
# dim.append(21)
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(dim, acc)
# ax.set_xlabel('Dimensions')
# ax.set_ylabel('Accuracy')
#
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1,1,1)
# ax1.plot(dim, timelist)
# ax1.set_xlabel('Dimensions')
# ax1.set_ylabel('Time')
# plt.show()
#
# # NN
# acc = []
# timelist = []
# dim = list(range(3, 21))
# for i in dim:
#     t1 = time()
#     acc.append(Neural_Network.main(dr=i))
#     t2 = time()
#     timelist.append(t2-t1)
#     print(t2-t1)
#
# t1 = time()
# acc.append(Neural_Network.main(dr=0))
# t2 = time()
# timelist.append(t2-t1)
# print(t2-t1)
# dim.append(21)
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(dim, acc)
# ax.set_xlabel('Dimensions')
# ax.set_ylabel('Accuracy')
#
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1,1,1)
# ax1.plot(dim, timelist)
# ax1.set_xlabel('Dimensions')
# ax1.set_ylabel('Time')
# plt.show()

# ===========================Deleted Data===========================
accuracy = Neural_Network.main(dr=16, md=True)
print(f"Accuracy: {accuracy}")