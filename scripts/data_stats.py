from context import expert_finding
import expert_finding.io
import logging

logger = logging.getLogger()

dataset_names = expert_finding.io.get_list_of_dataset_names()

print("Names of the datasets available:")
for dn in dataset_names:
    print()
    print(dn)
    A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset(dn)
    print("num candidates:", A_da.shape[1])
    print("num documents:", A_da.shape[0])
    print("num expertise fields:", len(tags))
    print("num experts:", L_a.shape[1])
    print("num queries:", L_d.shape[0])
    print(tags)
print()