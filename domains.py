import torch
from torch.nn import functional as F

DOMAIN_ENUMS = {
    0: "Wikipedia",
    1: "News articles",
    2: "Movie reviews",
    3: "Examinations"
}

DATASET_DOMAINS = {
    "squad": 0,
    "nat_questions": 0,
    "newsqa": 1,
    "duorc": 2,
    "race": 3,
    "relation_extraction": 0
}

DATASET_NUMBERS = {
    "squad": 50000,
    "nat_questions": 50000,
    "newsqa": 50000,
    "duorc": 127,
    "race": 127,
    "relation_extraction": 127
}

class_weights = [1., 1., 1., 1.]
sample_counts = [0, 0, 0, 0]
for dataset, n in DATASET_NUMBERS.items():
    domain = DATASET_DOMAINS[dataset]
    sample_counts[domain] += n
class_weights = [class_weights[i] / sample_counts[i] for i in range(len(class_weights))]


def domains_to_one_hot(domains, n=len(DOMAIN_ENUMS)):
    return F.one_hot(domains, num_classes=n)
