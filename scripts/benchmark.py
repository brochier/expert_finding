from context import expert_finding
import expert_finding.io
import expert_finding.evaluation
import expert_finding.models.random_model
import expert_finding.models.panoptic_model
import expert_finding.models.propagation_model
import expert_finding.models.voting_model
import expert_finding.models.voting_tadw_model
import expert_finding.models.propagation_tadw_model
import expert_finding.models.voting_idne_model
import expert_finding.models.propagation_idne_model
import expert_finding.models.pre_ane_model
import expert_finding.models.post_ane_model
import expert_finding.models.tadw
import expert_finding.models.gvnrt
import expert_finding.models.graph2gauss
import expert_finding.models.idne
import expert_finding.models.gvnrt_expert_model
import numexpr

import os, sys, resource
import logging

logger = logging.getLogger()

numexpr.set_num_threads(numexpr.detect_number_of_cores())

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.90, hard))

def main():
    dataset_names = expert_finding.io.get_list_of_dataset_names()
    #dataset_names = ['dblp', 'stats.stackexchange.com', 'academia.stackexchange.com']
    #dataset_names = ['mathoverflow.net']

    models = {
        #"random": expert_finding.models.random_model.Model(),
        #"panoptic": expert_finding.models.panoptic_model.Model(),
        #"voting": expert_finding.models.voting_model.Model(),
        #"propagation": expert_finding.models.propagation_model.Model(),
        #"voting tadw": expert_finding.models.voting_tadw_model.Model(),
        #"propagation tadw": expert_finding.models.propagation_tadw_model.Model(),
        #"voting (IDNE)": expert_finding.models.voting_idne_model.Model(),
        #"propagation (IDNE)": expert_finding.models.propagation_idne_model.Model(),
        #"pre idne": expert_finding.models.pre_ane_model.Model(expert_finding.models.idne.Model),
        #"post idne": expert_finding.models.post_ane_model.Model(expert_finding.models.idne.Model),
        "gvnrt expert": expert_finding.models.gvnrt_expert_model.Model(),
        #"pre tadw": expert_finding.models.pre_ane_model.Model(expert_finding.models.tadw.Model),
        #"pre gvnrt": expert_finding.models.pre_ane_model.Model(expert_finding.models.gvnrt.Model),
        #"pre g2g": expert_finding.models.pre_ane_model.Model(expert_finding.models.graph2gauss.Model),
        #"post tadw": expert_finding.models.post_ane_model.Model(expert_finding.models.tadw.Model),
        #"post gvnrt": expert_finding.models.post_ane_model.Model(expert_finding.models.gvnrt.Model),
        #"post g2g": expert_finding.models.post_ane_model.Model(expert_finding.models.graph2gauss.Model),

    }

    for dataset_name in dataset_names:
        A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = expert_finding.io.load_dataset(dataset_name)
        logger.info('')
        logger.info(f"{dataset_name:<30}{'& ROC AUC':<20}{'& P@10':<20}{'& AP':<20}{'& RR':<20}   \\\\")
        for model_name, model in models.items():
            eb = expert_finding.evaluation.run_all_evaluations(model, A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask)
            me = expert_finding.evaluation.merge_evaluations(eb, tags)
            logger.info(
                f"{model_name:<30}"
                f"& {me['metrics']['ROC AUC']*100:05.2f}+-{me['std']['ROC AUC']*100:05.2f}      "
                f"& {me['metrics']['P@10']*100:05.2f}+-{me['std']['P@10']*100:05.2f}      "
                f"& {me['metrics']['AP']*100:05.2f}+-{me['std']['AP']*100:05.2f}      "
                f"& {me['metrics']['RR']:05.2f}+-{me['std']['RR']:05.2f}         \\\\"
            )
        del A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags

if __name__ == '__main__':
    memory_limit()  # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)