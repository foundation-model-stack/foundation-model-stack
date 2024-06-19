import matplotlib.pyplot as plt
import random
import torch
from utils import random_rotation_almost_hadamard

k = 4096

r_rand, r_inv_rand = random_rotation_almost_hadamard(k, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=False)
r_inv_rand_test = r_rand.T
identity = torch.eye(k)
test_identity = r_rand.type(torch.float64) @ r_inv_rand_test.type(torch.float64)
identity_diff = torch.abs(test_identity - identity)
print("Max identity diff:", float(torch.max(identity_diff)))
print("mean identity diff:", float(torch.mean(identity_diff)))

r_rand_dots = []
r_inv_rand_dots = []

comparison_count = 10000

for _ in range(comparison_count):
    i, j = 0, 0
    while i == j:
        i, j = random.randrange(k), random.randrange(k)
    dot_prod = torch.nn.functional.cosine_similarity(r_rand[:, i], r_rand[:, j], dim=0)
    dot_prod_inv = torch.nn.functional.cosine_similarity(r_inv_rand_test[i], r_inv_rand_test[j], dim=0)
    r_rand_dots.append(dot_prod)
    r_inv_rand_dots.append(dot_prod_inv)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
n_bins = 64

axs[0].hist(r_rand_dots, bins=n_bins)
axs[1].hist(r_inv_rand_dots, bins=n_bins)

plt.show()