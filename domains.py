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

def domains_to_one_hot(domains, n=len(DOMAIN_ENUMS)):
    return F.one_hot(domains, num_classes=n)
