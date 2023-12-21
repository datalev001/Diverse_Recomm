
import numpy as np
def matrix_factorization_recommendation(user_item_matrix, latent_features, learning_rate=0.01, regularization_rate=0.1, iterations=100):
    # Step 1: User-Item Interaction Matrix
    R = np.array(user_item_matrix)

    # Step 2: Matrix Factorization
    m, n = R.shape
    U = np.random.rand(m, latent_features)
    V = np.random.rand(n, latent_features)

    # Step 3: Optimize the objective function
    for _ in range(iterations):
        for i in range(m):
            for j in range(n):
                if R[i, j] > 0:
                    error_ij = R[i, j] - np.dot(U[i, :], V[j, :].T)
                    U[i, :] += learning_rate * (2 * error_ij * V[j, :] - 2 * regularization_rate * U[i, :])
                    V[j, :] += learning_rate * (2 * error_ij * U[i, :] - 2 * regularization_rate * V[j, :])

    # Step 4: Train the recommendation model (no explicit output, U and V are updated in-place)
    # Step 5: Generate prediction matrix
    R_prime = np.dot(U, V.T)
    # Step 6: Generate recommendations
    def generate_recommendations(user_id):
        user_predictions = R_prime[user_id, :]
        recommended_item = np.argmax(user_predictions)
        return recommended_item

    return generate_recommendations

# Example Usage
user_item_matrix = [
    [4, 5, 0, 2],
    [0, 3, 4, 1],
    [5, 2, 1, 4],
    [0, 1, 5, 3],
]

latent_features = 2
generate_recommendations = matrix_factorization_recommendation(user_item_matrix, latent_features)


###############code2##########################
# rating_trans_df = dataset.copy()
# rating_trans_df = rep_forecast_df.copy()
# liked_thesh = 1
# liked_thesh = 3
# N_singulars = 4
# binary_flag = False
# binary_flag = True
# recom_n =3
# Items_SVD_Filter: matrix SVD item-user profiles Filter using above two rating data

def Items_SVD_Filter(rating_trans_df, setfile):
    
    recom_filter_dict = get_recom_filter_dict(setfile)
    vals = list(recom_filter_dict.values())
    
    category, model_type, cnt_threshold, liked_thesh,\
    N_singulars, binary_flag, recom_n, n_neigh,\
    based_n, date_range, trans_data, trans_date, timeformat = vals
        
    user_rating_pivot = rating_trans_df.pivot(index= 'user_id', columns= 'item_id',\
                              values= 'rating').reset_index()       
    
    item_set = set(rating_trans_df['item_id'])

    user_rating_pivot = user_rating_pivot.fillna(0)
    
    users_df = user_rating_pivot[['user_id']]
    users_df['user_seq'] = range(1, len(users_df) + 1) 
    users = list(users_df['user_seq'])
    
    rating_cols = list(user_rating_pivot.columns)
    rating_cols = remove_sub_list(rating_cols, ['user_id', 'item_id'], False)
    matrix_rating_pivot = user_rating_pivot[rating_cols].values
    
    # range std for matrix_rating_pivot
    range_col = matrix_rating_pivot.max(axis=0) - matrix_rating_pivot.min(axis=0)
    matrix_rating_pivot = (matrix_rating_pivot - matrix_rating_pivot.min(axis=0)) / range_col
    matrix_rating_pivot.shape

    M = np.mean(matrix_rating_pivot, axis = 0)
    passv = 1
    liked_items_lst = []
    for i in range(len(matrix_rating_pivot)):
        liked_items = []  
        for j in range(len(matrix_rating_pivot[0])):
            if matrix_rating_pivot[i][j] > (M[j]*liked_thesh): 
                liked_items.extend([j+1])
                if binary_flag: matrix_rating_pivot[i][j] = 1
            else:    
                if binary_flag: matrix_rating_pivot[i][j] = 0
                passv = 1
                
        if len(liked_items)> 0: 
            liked_items_vector = [users[i], liked_items]
            liked_items_lst.append(liked_items_vector)

    liked_items_df = pd.DataFrame(liked_items_lst, columns = ['user_seq', 'item_id'])
    liked_items_df = liked_items_df.sort_values(['user_seq'])
    liked_items_df.shape
    
    U, sigma, Vt = svds(matrix_rating_pivot, k = N_singulars)
    
    sigma = np.diag(sigma)
    
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
    
    preds_df = pd.DataFrame(predicted_ratings, columns = rating_cols)
    preds_df.shape
    
    sorted_index = np.argsort(preds_df.to_numpy() * -1, axis=1)
    LL = sorted_index.shape[1] + 1
    rank_df = pd.DataFrame(preds_df.columns[sorted_index], index=preds_df.index, columns=range(1, LL)) 
    pred_with_rank = preds_df.join(rank_df.add_prefix('Rank'))
    
    rank_cols = ['Rank'+str(j) for j in range(1, len(item_set)+1)]
    preds_rank_df = pred_with_rank[rank_cols].reset_index()
    preds_rank_df['user_seq'] = preds_rank_df['index'] + 1
    preds_rank_df = preds_rank_df.drop(['index'], axis = 1)
    sel_ranks = ['Rank'+str(j) for j in range(1, recom_n+1)]
    preds_rank_df = preds_rank_df[['user_seq'] + sel_ranks]
    
    result_df = pd.merge(liked_items_df, preds_rank_df, on = ['user_seq'], how = 'inner')
    result_df = pd.merge(result_df, users_df, on = ['user_seq'], how = 'inner')
    
    result_df.shape
    result_df.head(200)
    
    return result_df




