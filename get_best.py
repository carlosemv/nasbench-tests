import numpy as np
from nasbench import api


nasbench = api.NASBench('nasbench_full.tfrecord')

best = {e: (None, None, 0) for e in nasbench.valid_epochs}
for i, h in enumerate(nasbench.hash_iterator()):
    model, stats = nasbench.get_metrics_from_hash(h)
    for epochs in nasbench.valid_epochs:
            acc = np.mean([sample['final_test_accuracy'] for sample in stats[epochs]])
            if acc > best[epochs][-1]:
                    best[epochs] = (model, stats, acc)

np.save('best_file.npy', best)
# best = np.load('best_file.npy', allow_pickle=True).item()