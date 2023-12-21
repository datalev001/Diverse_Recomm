
import numpy as np

def incremental_matrix_factorization(num_users, num_items, latent_factors, user_id, item_id, eta=0.01):
    def initialize_matrices(num_users, num_items, latent_factors):
        U = np.random.rand(num_users, latent_factors)
        V = np.random.rand(latent_factors, num_items)
        return U, V

    def update_matrices(U, V, user_id, item_id, eta):
        U[user_id, :] += eta * ((V @ V.T)[user_id, :] - U[user_id, :] @ V[item_id, :].reshape(-1, 1) @ V[item_id, :].reshape(1, -1))
        V[:, item_id] += eta * ((U.T @ U)[:, item_id] - V[:, item_id] @ U[user_id, :].reshape(-1, 1) @ U[user_id, :].reshape(1, -1))

    def make_recommendation(U, V, user_id, item_id):
        return U[user_id, :] @ V[:, item_id]

    U, V = initialize_matrices(num_users, num_items, latent_factors)
    update_matrices(U, V, user_id, item_id, eta)
    recommendation_score = make_recommendation(U, V, user_id, item_id)

    return U, V, recommendation_score

# Example Usage with Suitable Parameters
num_users = 100
num_items = 50
latent_factors = 10
user_id = 5
item_id = 10

U, V, recommendation_score = incremental_matrix_factorization(num_users, num_items, latent_factors, user_id, item_id)

print("Updated U Matrix:")
print(U)
print("\nUpdated V Matrix:")
print(V)
print("\nRecommendation Score:")
print(recommendation_score)
