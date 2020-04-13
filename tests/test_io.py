from context import expert_finding
import expert_finding.io
import logging

logger = logging.getLogger()

dataset_names = expert_finding.io.get_list_of_dataset_names()

print("Names of the datasets available:")
for dn in dataset_names:
    print(dn)
print()

A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("stats.stackexchange.com")

print("Random Document:")
print(T[0])
print()

print("Random Tag:")
print(tags[0])
print()
