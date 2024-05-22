from rinarak.domain import load_domain_string, Domain
from rinarak.knowledge.executor import CentralExecutor
domain_parser = Domain("base.grammar")
from dataclasses import dataclass
from primitives import *
import sys

meta_domain_str = ""
with open(f"domains/demo_domain.txt","r") as domain:
        for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)


executor = CentralExecutor(domain, "cone", 100)
for predicate in executor.predicates[1]:
      refractor(executor, predicate.name)


from rinarak.algs.search import run_heuristic_search
import random

num = 4
rand_end = torch.ones([num])
rand_feat = torch.randn([num,100])

context = {
        "end": logit(rand_end),
        "pos": torch.randn([4,3]),
        "red": -8*torch.ones([4]), #{"end": torch.ones([4])},,
        "green": -8*torch.ones([4]),
        "movable":  logit(torch.tensor([1.0, 1.0, 0.0, 1.0])).reshape([num]),
        "moving":  -8*torch.ones([num]),
        "features": rand_feat,
        "left": torch.ones([4,4]),
        "executor": executor#{"end": torch.zeros([4])}
    }



#unique_at_operator = Primitive("unique_at", arrow(ObjectSet, Integer, Boolean),)

def unique_at(searchState, k = 0):
    
    probabilities = torch.min(searchState.state["end"], searchState.state["moving"]).sigmoid()
    
    #print(probabilities)
    #probabilities = torch.tensor([-8, 8, -8])
    p1 =  probabilities[k]
    #probabilities[k] = 0.0
    
    
    p_only_one = p1 # torch.min((probabilities < 0.5)).float()
    return p_only_one > 0.5 and torch.all(probabilities[:k] < 0.5) and torch.all(probabilities[k+1:] < 0.5)

def get_unique_at(idx = 0):
      return lambda x: unique_at(x, idx)


print("start the goal search")

print(context["movable"].cpu().detach().numpy())
print(context["moving"].cpu().detach().numpy())
print(context["end"].cpu().detach().numpy())

#_, demo_execute = executor.apply_action("make_motion",[0], context)

#print(demo_execute["moving"])

#state, action, costs, nr = executor.search_discrete_state(context, "(exists (moving $0) )")
state, action, costs, nr = executor.search_discrete_state(context, get_unique_at(3))
#print(state[-1].state["movable"].cpu().detach().numpy())
print(state[-1].state["moving"].cpu().detach().numpy())
#print(context["end"].cpu().detach().numpy())
print()
print(action)
print(costs)
print(nr)
