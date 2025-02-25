# import numpy as np
#
# v = np.array([1, 3.9, -9, 2])
# print(v, v.ndim)





# import numpy as np
# import pandas as pd
#
# df = pd.DataFrame(
#     {"a" : [4, 5, 6],
#           "b" : [7, 8, 9],
#           "c" : [10, 11, 12]}, index = [1, 2, 3]
# )
# print(df)





# import numpy as np
# import pandas as pd
#
# df = pd.DataFrame(
#     [[4, 7, 10],
#      [5, 8, 11],
#      [6, 9, 12]], index=[1, 2, 3], columns=['a', 'b', 'c']
# )
# print(df)






# #from statistics import LinearRegression
# from sklearn.linear_model import LinearRegression
# # from sklearn.neighbors import KNeighborsRegressor
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
# #print(type(ls))
# #print(ls)
# X = ls[["GDP per capita (USD)"]].values
# y = ls[["Life satisfaction"]].values
# #print(X)
#
# # ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
# # plt.axis([23500, 62500, 4, 9])
# # plt.show()
#
# model = LinearRegression()
# # model = KNeighborsRegressor(n_neighbors=3)
# model.fit(X, y)  # 머신 러닝
#
# X_new = [[31721.3]]  # ROK 2020
# print(model.predict(X_new))
# # LinearRegression 5.90
# # KNeighborsRegressor 5.70
