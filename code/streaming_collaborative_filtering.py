import numpy as np

def streaming_collaborative_filtering(num_users, num_items, latent_factors, user_id, item_id, rating, eta=0.01, lambd=0.01):
    U = np.random.rand(num_users, latent_factors)
    V = np.random.rand(latent_factors, num_items)

    def update_matrices_sgd(U, V, user_id, item_id, rating, eta, lambd):
        error_ui = rating - U[user_id, :] @ V[:, item_id]
        U[user_id, :] += eta * (error_ui * V[:, item_id] - lambd * U[user_id, :])
        V[:, item_id] += eta * (error_ui * U[user_id, :] - lambd * V[:, item_id])

    def make_recommendation(U, V, user_id, item_id):
        return U[user_id, :] @ V[:, item_id]

    update_matrices_sgd(U, V, user_id, item_id, rating, eta, lambd)
    recommendation_score = make_recommendation(U, V, user_id, item_id)

    return U, V, recommendation_score

# Example Usage with Suitable Parameters
num_users = 100
num_items = 50
latent_factors = 10
user_id = 5
item_id = 10
rating = 4  # Assume a rating or relevance score

U, V, recommendation_score = streaming_collaborative_filtering(num_users, num_items, latent_factors, user_id, item_id, rating)

print("Updated U Matrix:")
print(U)
print("\nUpdated V Matrix:")
print(V)
print("\nRecommendation Score:")
print(recommendation_score)
