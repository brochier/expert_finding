from context import expert_finding
import expert_finding.io
import expert_finding.evaluation
import expert_finding.models.random_model
import expert_finding.models.panoptic_model
import expert_finding.models.voting_model
import expert_finding.models.propagation_model
import expert_finding.models.voting_idne_model
import expert_finding.models.propagation_idne_model
import expert_finding.models.gvnrt_expert_model
import expert_finding.models.pre_ane_model
import expert_finding.models.post_ane_model
import expert_finding.models.tadw
import expert_finding.models.gvnrt
import expert_finding.models.idne
import expert_finding.models.graph2gauss
import numpy as np
import logging

logger = logging.getLogger()

#A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("stats.stackexchange.com")
A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset("academia.stackexchange.com")

#model = expert_finding.models.pre_ane_model.Model(expert_finding.models.idne.Model)
model = expert_finding.models.propagation_idne_model.Model()

eval_batches, merged_eval = expert_finding.evaluation.run(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags)
