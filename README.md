## **expert_finding**: Python package for BIR@ECIR2020.

### Install:

You need python 3.7 installed on your computer with pip. Optionally create a new environment (with conda):
    
    conda create --name expert_finding python=3.7 pip
    conda activate expert_finding
    
Then run:

    pip install git+https://github.com/brochier/expert_finding
    
Or:

    git clone https://github.com/brochier/expert_finding
    cd expert_finding
    pip install -r requirements.txt  

### Example script

```python
"""
An example script to use expert_finding as a package. Its shows how to load a dataset, create a model and run an evaluation.
"""

import expert_finding.io
import expert_finding.evaluation
import expert_finding.models.random_model
import expert_finding.models.panoptic_model
import expert_finding.models.propagation_model
import expert_finding.models.voting_model
import numpy as np


# Print the list of available datasets
dataset_names = expert_finding.io.get_list_of_dataset_names()
print("Names of the datasets available:")
for dn in dataset_names:
    print(dn)
print()


# Load one dataset
A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("stats.stackexchange.com")


"""
A_da : adjacency matrix of the document-candidate network (scipy.sparse.csr_matrix)
A_dd : adjacency matrix of the document-document network (scipy.sparse.csr_matrix)
T : raw textual content of the documents (numpy.array)
L_d : labels associated to the document (corresponding to T[L_d_mask]) (numpy.array)
L_d_mask : mask to select the labeled documents (numpy.array)
L_a : labels associated to the candidates (corresponding to A_da[:,L_d_mask]) (numpy.array)
L_a_mask : mask to select the labeled candidates (numpy.array)
tags : names of the labels of expertise (numpy.array)
"""

# You can load a model
#model = expert_finding.models.panoptic_model.Model()
#model = expert_finding.models.voting_model.Model()
#model = expert_finding.models.propagation_model.Model()


# You can create  a model
class Model:
    def __init__(self):
        self.num_candidates = 0

    def fit(self, A_da, A_dd, T):
        self.num_candidates = A_da.shape[1]

    def predict(self, d, mask = None):
        if mask is not None:
            self.num_candidates = len(mask)
        return np.random.rand(self.num_candidates)

model = Model()

# Run an evaluation
eval_batches, merged_eval = expert_finding.evaluation.run(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags)


# This last function actually performs 3 sub functions:

# 1) run all available querries and compute the metrics for each of them
eval_batches = expert_finding.evaluation.run_all_evaluations(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask)

# 2) Merge the evaluations by averaging over the metrics
merged_eval = expert_finding.evaluation.merge_evaluations(eval_batches, tags)

#3) Plot the evaluation. If path is not None, the plot is not shown but saved in an image on disk. 
expert_finding.evaluation.plot_evaluation(merged_eval, path=None)
```
    

 
 