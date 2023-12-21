#############code 1: assuming  user_item_matrix is given, using matrix operation***********
import numpy as np
def item_based_collaborative_filtering(user_item_matrix):

    U = np.array(user_item_matrix)
    U_prime = U - np.mean(U, axis=1, keepdims=True)
    S = np.dot(U_prime, U_prime.T) / (np.linalg.norm(U_prime, axis=1, keepdims=True) @ np.linalg.norm(U_prime, axis=1, keepdims=True).T)
    P = np.dot(S, U_prime) / np.abs(S).sum(axis=1, keepdims=True)
    P_prime = P + np.mean(U, axis=1, keepdims=True)
    recommendations = np.argmax(P_prime, axis=1)

    return recommendations

# Example Usage
user_item_matrix = [
    [4, 5, 0, 2],
    [0, 3, 4, 1],
    [5, 2, 1, 4],
    [0, 1, 5, 3],
]

user_recommendations = item_based_collaborative_filtering(user_item_matrix)
print("User Recommendations:", user_recommendations)


##########################code 2#######################################
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def Collaborative_Filtering(trans_df, user_id, product_id, qty):
    # Step 1: Generate user-item interaction matrix
    user_item_matrix = trans_df.pivot_table(index=user_id, columns=product_id, values=qty, fill_value=0)

    # Step 2: Perform matrix normalization
    user_item_mean = user_item_matrix.mean(axis=1)
    user_item_normalized = user_item_matrix.sub(user_item_mean, axis=0)

    # Step 3: Perform item similarity calculation
    item_similarity_matrix = pd.DataFrame(cosine_similarity(user_item_normalized.T), index=user_item_matrix.columns, columns=user_item_matrix.columns)

    # Step 4: Perform user-item interaction predictions
    predicted_matrix = pd.DataFrame(index=user_item_normalized.index, columns=user_item_normalized.columns)

    for i in user_item_normalized.index:
        for j in user_item_normalized.columns:
            numerator = sum(item_similarity_matrix.loc[j, k] * user_item_normalized.loc[i, k] for k in user_item_normalized.columns)
            denominator = sum(abs(item_similarity_matrix.loc[j, k]) for k in user_item_normalized.columns)
            predicted_matrix.loc[i, j] = numerator / denominator if denominator != 0 else 0

    # Step 5: Make de-normalization predictions
    predicted_denormalized = predicted_matrix.add(user_item_mean, axis=0)

    # Step 6: Generate recommendations
    user_recommendations = pd.DataFrame(index=predicted_denormalized.index, columns=['Recommended_Product'])
    
    for user in predicted_denormalized.index:
        recommended_product = predicted_denormalized.loc[user].idxmax()
        user_recommendations.loc[user, 'Recommended_Product'] = recommended_product

    return user_recommendations

# Example usage:
# user_recommendations = Collaborative_Filtering(trans_df, 'user_id', 'product_id', 'qty')
# print(user_recommendations)


#############code 3: using knn.kneighbors*********************************************

def Items_Collaborative_Filter(rating_trans_df, user_id, item_id, rating):
    
    def remove_sub_list(lst, sub_lst, dup):
    if dup:
        filter_set = set(sub_lst)
        lst_fixed = [x for x in lst if x not in filter_set]
    else:    
        lst_fixed = list(set(lst) - set(sub_lst))
        
    return lst_fixed 

    
    liked_thesh = 3
    binary_flag = 1 
    n_neigh = 3
    based_n = 2
                
    # pivot rating_trans_df to make it as user -- rating maxtrix for calculating cos distance      
    user_rating_pivot = rating_trans_df.pivot(index= user_id, columns= item_id,\
                              values= rating).reset_index()       
           
    user_rating_pivot = user_rating_pivot.fillna(0)
    users = list(user_rating_pivot[user_id])
    
    item_set = set(rating_trans_df[item_id])
        
    rating_cols = list(user_rating_pivot.columns)
    rating_cols = remove_sub_list(rating_cols, [user_id, item_id], False)
    matrix_rating_pivot = user_rating_pivot[rating_cols].astype(float).values
    
    # range std for matrix_rating_pivot
    range_col = matrix_rating_pivot.max(axis=0) - matrix_rating_pivot.min(axis=0)
    matrix_rating_pivot = (matrix_rating_pivot - matrix_rating_pivot.min(axis=0)) / range_col
    matrix_rating_pivot.shape
          
    M = matrix_rating_pivot.mean(axis = 0)
    passed = 1          
    
    liked_items_lst = []
    for i in range(len(matrix_rating_pivot)):
        liked_items = []  
        for j in range(len(matrix_rating_pivot[0])):
            if matrix_rating_pivot[i][j] > (M[j]*liked_thesh): 
                if binary_flag: matrix_rating_pivot[i][j] = 1
                liked_items.extend([j+1])
            else:
                if binary_flag: matrix_rating_pivot[i][j] = 0
                passed = 1
                
        if len(liked_items)> 0: 
            liked_items_vector = [users[i], liked_items]
            liked_items_lst.append(liked_items_vector)

    liked_items_df = pd.DataFrame(liked_items_lst, columns = [user_id, item_id])
    liked_items_df = liked_items_df.sort_values([user_id])
    liked_items_df.shape
    
    # place matrix_rating_pivot into sparse matrix and cal cos distance using knn               
    matrix_rating_sparse = csr_matrix(matrix_rating_pivot)
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
    knn.fit(matrix_rating_sparse)

    recom_items_lst = []
    iter_num = 1
    for it in  liked_items_lst:  
            
        user_id_sel = it[0]
        
        based_on_items = liked_items_df[liked_items_df[user_id] == user_id_sel][item_id].values[0]
        # recommand based on 3 items liked
        if len(based_on_items) > based_n:
            based_on_items = based_on_items[:based_n]
        
        recomm_indices = []
        
        if len(based_on_items) > 0:  
            for i in based_on_items:
                distances , indices = knn.kneighbors(matrix_rating_sparse[i], n_neighbors = n_neigh)
                indices = indices.flatten()
                sel_indices= indices[1:]
                recomm_indices.extend(sel_indices)
        
        recom_lst = list(set(recomm_indices))
        recom_lst = [ity + 1 for ity in recom_lst]
        recom_set = set(recom_lst) & item_set
        recom_lst = [ity for ity in recom_set]
        
        recom_lst_new = list(set(recom_lst) - set(based_on_items))
        if len(recom_lst) > 0:
            recom_lst = sorted(recom_lst)
            recom_lst_new = sorted(recom_lst_new)
            recom_items_lst.append([user_id_sel, based_on_items, recom_lst, recom_lst_new])
        iter_num = iter_num + 1
        if iter_num > 100: break
            
    result_df = pd.DataFrame(recom_items_lst, columns = [user_id, liked_items, recom_items, recom_new_items])
    
    # result_df.head(300)
       
    return result_df


