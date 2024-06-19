import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def main():
    # 데이터 읽기
    df1 = pd.read_csv('C:/Users/user/ratings.dat', delimiter='::', header=None, names=['User', 'Item', 'rating', 'timestamp'], engine='python')

    num_users = df1["User"].max()
    num_items = df1['Item'].max()
    
    ratings_matrix = np.zeros((num_users, num_items))
    for row in df1.itertuples():
        ratings_matrix[row.User - 1, row.Item - 1] = row.rating
    ratings_matrix = np.matrix(ratings_matrix)
    
    # KMeans 클러스터링
    ratings_matrix = pd.DataFrame(ratings_matrix)
    km = KMeans(n_clusters=3, random_state=0)
    km.fit(ratings_matrix)
    y_km = km.predict(ratings_matrix)
    print(y_km)
    
    ratings_matrix['label'] = y_km
    ratings_matrix_0 = ratings_matrix[ratings_matrix['label'] == 0]
    ratings_matrix_1 = ratings_matrix[ratings_matrix['label'] == 1]
    ratings_matrix_2 = ratings_matrix[ratings_matrix['label'] == 2]
    print("Cluster 0:\n", ratings_matrix_0)
    print("Cluster 1:\n", ratings_matrix_1)
    print("Cluster 2:\n", ratings_matrix_2)
    
 # AU 알고리즘 적용
    AU_0 = np.sum(ratings_matrix_0, axis=0)
    AU_1 = np.sum(ratings_matrix_1, axis=0)
    AU_2 = np.sum(ratings_matrix_2, axis=0)

    data_0 = {'AU': AU_0}
    df_0 = pd.DataFrame(data_0)
    data_1 = {'AU': AU_1}
    df_1 = pd.DataFrame(data_1)
    data_2 = {'AU': AU_2}
    df_2 = pd.DataFrame(data_2)
    top10_AU_0 = df_0.sort_values(by='AU',ascending = False).head(10)
    top10_AU_1 = df_1.sort_values(by='AU',ascending = False).head(10)
    top10_AU_2 = df_2.sort_values(by='AU',ascending = False).head(10)

    # 결과 출력
    print("\nAU알고리즘 결과:")
    print("Cluster 0:\n", top10_AU_0) # Clustering0 그룹의 top10
    print("Cluster 1:\n", top10_AU_1) # Clustering1 그룹의 top10
    print("Cluster 2:\n", top10_AU_2) # Clustering2 그룹의 top10

    # Average (Avg) 알고리즘
    l_0 = len(ratings_matrix_0['label'])
    avg_0 = np.zeros(l_0)
    for i in range(l_0):
        a = sum(ratings_matrix_0.iloc[i,:] != 0) - 1 
        m = sum(ratings_matrix_0.iloc[i,:]) / a
        avg_0[i] = m

    ratings_matrix_0_a = ratings_matrix_0.loc[:,'label']
    ratings_matrix_0_a = pd.DataFrame(ratings_matrix_0_a)
    ratings_matrix_0_a['AVG'] = avg_0
    ratings_matrix_0_a

    l_1 = len(ratings_matrix_1['label'])
    avg_1 = np.zeros(l_1)
    for i in range(l_1):
        a = sum(ratings_matrix_1.iloc[i,:] != 0) - 1 
        m = sum(ratings_matrix_1.iloc[i,:]) / a
        avg_1[i] = m

    ratings_matrix_1_a = ratings_matrix_1.loc[:,'label']
    ratings_matrix_1_a = pd.DataFrame(ratings_matrix_1_a)
    ratings_matrix_1_a['AVG'] = avg_1
    ratings_matrix_1_a

    l_2 = len(ratings_matrix_2['label'])
    avg_2 = np.zeros(l_2)
    for i in range(l_2):
        a = sum(ratings_matrix_2.iloc[i,:] != 0) - 1 
        m = sum(ratings_matrix_2.iloc[i,:]) / a
        avg_2[i] = m

    ratings_matrix_2_a = ratings_matrix_2.loc[:,'label']
    ratings_matrix_2_a = pd.DataFrame(ratings_matrix_2_a)
    ratings_matrix_2_a['AVG'] = avg_2
    ratings_matrix_2_a

    ratings_matrix_0_a = ratings_matrix_0_a.sort_values(by='AVG', ascending=False)
    ratings_matrix_0_a_top10 = ratings_matrix_0_a.head(10)
    ratings_matrix_1_a = ratings_matrix_1_a.sort_values(by='AVG', ascending=False)
    ratings_matrix_1_a_top10 = ratings_matrix_1_a.head(10)
    ratings_matrix_2_a = ratings_matrix_2_a.sort_values(by='AVG', ascending=False)
    ratings_matrix_2_a_top10 = ratings_matrix_2_a.head(10)

    print("\nAvg 결과:")
    print("Cluster 0:\n", ratings_matrix_0_a_top10) # Clustering0 그룹의 top10
    print("Cluster 1:\n", ratings_matrix_1_a_top10) # Clustering1 그룹의 top10
    print("Cluster 2:\n", ratings_matrix_2_a_top10) # Clustering2 그룹의 top10

    # SC알고리즘
    non_zero_0 = np.count_nonzero(ratings_matrix_0, axis=0)
    non_zero_1 = np.count_nonzero(ratings_matrix_1, axis=0)
    non_zero_2 = np.count_nonzero(ratings_matrix_2, axis=0)
    
    non_zero_df0 = pd.DataFrame({
        'Item': range(0, len(non_zero_0) ),
        'NonZeroCount': non_zero_0
        })
    non_zero_df0 = non_zero_df0.sort_values(by='NonZeroCount', ascending=False)
    non_zero_df0  = non_zero_df0.head(10)

    non_zero_df1 = pd.DataFrame({
        'Item': range(0, len(non_zero_1) ),
        'NonZeroCount': non_zero_1
        })
    non_zero_df1 = non_zero_df1.sort_values(by='NonZeroCount', ascending=False)
    non_zero_df1  = non_zero_df1.head(10)

    non_zero_df2 = pd.DataFrame({
        'Item': range(0, len(non_zero_2) ),
        'NonZeroCount': non_zero_2
        })
    non_zero_df2 = non_zero_df2.sort_values(by='NonZeroCount', ascending=False)
    non_zero_df2  = non_zero_df2.head(10)

    print("\nSC 알고리즘 결과:")
    print("Cluster 0:\n", non_zero_df0) # Clustering0 그룹의 top10
    print("Cluster 1:\n", non_zero_df1) # Clustering1 그룹의 top10
    print("Cluster 2:\n", non_zero_df2) # Clustering2 그룹의 top10

    # AV 알고리즘
    over4_0 = np.sum(ratings_matrix_0 > 4, axis=0)
    over4df_0 = pd.DataFrame({
        'Item': range(len(over4_0)),
        'CountAbove4': over4_0
        })
    over4df_0 = over4df_0.sort_values(by='CountAbove4', ascending=False)
    over4df_0 = over4df_0.head(10)


    over4_1 = np.sum(ratings_matrix_1 > 4, axis=0)
    over4df_1 = pd.DataFrame({
        'Item': range(len(over4_1)),
        'CountAbove4': over4_1
        })
    over4df_1 = over4df_1.sort_values(by='CountAbove4', ascending=False)
    over4df_1 = over4df_1.head(10)

    over4_2 = np.sum(ratings_matrix_2 > 4, axis=0)
    over4df_2 = pd.DataFrame({
        'Item': range(len(over4_2)),
        'CountAbove4': over4_2
        })
    over4df_2 = over4df_2.sort_values(by='CountAbove4', ascending=False)
    over4df_2 = over4df_2.head(10)

    print("\nAV 알고리즘 결과:")
    print("Cluster 0:\n", over4df_0) # Clustering0 그룹의 top10
    print("Cluster 1:\n", over4df_1) # Clustering1 그룹의 top10
    print("Cluster 2:\n", over4df_2) # Clustering2 그룹의 top10

    # BC 알고리즘
    mat_0 = pd.DataFrame(ratings_matrix_0)
    mat_0= mat_0[mat_0 != 0]
    mat_r_0 = mat_0.rank(axis='columns')
    mat_r_0 = mat_r_0.mean(axis=1)
    mat_r_0 = mat_r_0.sort_values(ascending=False)
    mat_r_0 = mat_r_0.head(10)

    mat_1 = pd.DataFrame(ratings_matrix_1)
    mat_1= mat_1[mat_1 != 0]
    mat_r_1 = mat_1.rank(axis='columns')
    mat_r_1 = mat_r_1.mean(axis=1)
    mat_r_1 = mat_r_1.sort_values(ascending=False)
    mat_r_1 = mat_r_1.head(10)

    mat_2 = pd.DataFrame(ratings_matrix_2)
    mat_2= mat_2[mat_2 != 0]
    mat_r_2 = mat_2.rank(axis='columns')
    mat_r_2 = mat_r_2.mean(axis=1)
    mat_r_2 = mat_r_2.sort_values(ascending=False)
    mat_r_2 = mat_r_2.head(10)

    print("\nBC 알고리즘 결과:")
    print("Cluster 0:\n", mat_r_0) # Clustering0 그룹의 top10
    print("Cluster 1:\n", mat_r_1) # Clustering1 그룹의 top10
    print("Cluster 2:\n", mat_r_2) # Clustering2 그룹의 top10

    # CR 알고리즘
    score_j_0 = 0
    score_i_0 = 0
    num_0 = ratings_matrix_0.shape[1]
    scores_0 = np.zeros(num_0)
    for j in range(num_0):
        for i in range(j+1,num_0-0):
            if j != i:
                z_0 = ratings_matrix_0.iloc[:,j]
                u_0 = ratings_matrix_0.iloc[:,i]
                win_0 = (z_0>u_0).sum()
                lose_0 = (z_0<u_0).sum()
                j_sum_0 = win_0 - lose_0
                if(j_sum_0 > 0):
                    j_sum_0 = 1
                elif(j_sum_0 <0):
                   j_sum_0 = -1
                else:
                   j_sum_0=0
                i_sum_0 = -j_sum_0
                scores_0[j] += j_sum_0
                scores_0[i] += i_sum_0
            
    data_0 = {"index" : range(len(scores_0)), "scores" : scores_0}
    data_0 = pd.DataFrame(data_0)
    data_0 = data_0.sort_values(by = "scores", ascending=False)
    data_0 = data_0.head(10)

    score_j_1 = 0
    score_i_1 = 0    
    num_1 = ratings_matrix_1.shape[1]
    scores_1 = np.zeros(num_1)
    for j in range(num_1):
       for i in range(num_1):
          if j != i:
            z_1 = ratings_matrix_1.iloc[:,j]
            u_1 = ratings_matrix_1.iloc[:,i]
            win_1 = (z_1>u_1).sum()
            lose_1 = (z_1<u_1).sum()
            j_sum_1 = win_1 - lose_1
            if(j_sum_1 > 0):
               j_sum_1 = 1
            elif(j_sum_1 <0):
               j_sum_1 = -1
            else:
               j_sum_1=0
            i_sum_1 = -j_sum_1
            scores_1[j] += j_sum_1
            scores_1[i] += i_sum_1

    data_1 = {"index" : range(len(scores_1)), "scores" : scores_1}
    data_1 = pd.DataFrame(data_1)
    data_1 = data_1.sort_values(by = "scores", ascending=False)
    data_1 = data_1.head(10)

    score_j = 0
    score_i = 0
    num = ratings_matrix_2.shape[1]
    scores_2 = np.zeros(num)
    for j in range(num):
       for i in range(j+1,num):
          if j != i:
            z = ratings_matrix_2.iloc[:,j]
            u = ratings_matrix_2.iloc[:,i]
            win = (z>u).sum()
            lose = (z<u).sum()
            j_sum = win - lose
            if(j_sum > 0):
               j_sum = 1
            elif(j_sum <0):
               j_sum = -1
            else:
               j_sum=0
            i_sum = -j_sum
            scores_2[j] += j_sum
            scores_2[i] += i_sum 

    data_2 = {"index" : range(len(scores_2)), "scores" : scores_2}
    data_2 = pd.DataFrame(data_2)
    data_2 = data_2.sort_values(by = "scores", ascending=False)
    data_2 = data_2.head(10)

    print("\nCR 알고리즘 결과:")
    print("Cluster 0:\n", data_0) # Clustering0 그룹의 top10
    print("Cluster 1:\n", data_1) # Clustering1 그룹의 top10
    print("Cluster 2:\n", data_2) # Clustering2 그룹의 top10

if __name__ == "__main__":
    main()