import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .embedding  import build_box_registry
from .entailment import build_entailment
from .predicates import PredicateFilter
from rinarak.utils import freeze
from rinarak.utils.misc import *
from rinarak.utils.tensor import logit, expat
from rinarak.types import baseType, arrow
from rinarak.program import Primitive, Program
from rinarak.dsl.vqa_types import Boolean
from rinarak.algs.search.heuristic_search import run_heuristic_search
from dataclasses import dataclass
import copy
from itertools import combinations
import random

class UnknownArgument(Exception):
    def __init__(self):super()

class UnknownConceptError(Exception):
    def __init__(self):super()

@dataclass
class QuantizeTensorState(object):
      state: dict

    
def get_params(ps, token):
    start_loc = ps.index(token)
    ps = ps[start_loc:]
    count = 0
    outputs = ""
    idx = len(token) + 1
    while count >= 0:
         if ps[idx] == "(": count += 1
         if ps[idx] == ")": count -= 1
         outputs += ps[idx]
         idx += 1
    outputs = outputs[:-1]
    end_loc = idx + start_loc - 1
    return outputs.split(" "), start_loc, end_loc

class CentralExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, domain, concept_type = "cone", concept_dim = 100):
        super().__init__()
        BIG_NUMBER = 100
        entries = 64

        self.entailment = build_entailment(concept_type, concept_dim)
        self.concept_registry = build_box_registry(concept_type, concept_dim, entries)

        # [Types]
        self.types = domain.types
        for type_name in domain.types:
            baseType(type_name)

        # [Predicate Type Constraints]
        self.type_constraints = domain.type_constraints

        # [Predicates]
        self.predicates = {}
        for predicate in domain.predicates:
            predicate_bind = domain.predicates[predicate]
            predicate_name = predicate_bind["name"]
            params = predicate_bind["parameters"]
            rtype = predicate_bind["type"]

            # check the arity of the predicate
            arity = len(params)

            if arity not in self.predicates:
                self.predicates[arity] = []
            
            #predicate_imp = PredicateFilter(predicate_name,arity)
            self.predicates[arity].append(Primitive(predicate_name,arrow(Boolean, Boolean),predicate_name))
        
        # [Derived]
        self.derived = domain.derived

        # [Actions]
        self.actions = domain.actions

        # [Word Vocab]
        #self.relation_encoder = nn.Linear(config.object_dim * 2, config.object_dim)

        self.concept_vocab = []
        for arity in self.predicates:
            for predicate in self.predicates[arity]:
                self.concept_vocab.append(predicate.name)

        """Neuro Component Implementation Registry"""
        self.implement_registry = {}
        for implement_key in domain.implementations:
            effect = domain.implementations[implement_key]
            self.implement_registry[implement_key] = Primitive(implement_key,arrow(Boolean,Boolean),effect)

        # copy the implementations from the registry

        # args during the execution
        self.kwargs = None 

        self.effective_level = BIG_NUMBER

        self.quantized = False
    
    def check_implementation(self):
        warning = False
        for key in self.implement_registry:
            func_call = self.implement_registry[key]
            if func_call is None:warning = True
        if warning:
            print("Warning: exists predicates not implemented.")
            return False
    
    def redefine_predicate(self, name, func):
        for predicate in Primitive.GLOBALS:
            if predicate== name:
                Primitive.GLOBALS[name].value = func
        return True
 
    def evaluate(self, program, context):
        """program as a string to evaluate under the context
        Args:
            program: a string representing the expression for evaluation
            context: the execution context with predicates and executor
        Return:
            precond: a probability of this action is successfully activated.
            parameters changed
        """
        BIG_NUM = 1e5
        flat_string = program
        flag = True in [derive in flat_string for derive in self.derived]
        itr = 0
        """Replace all the derived expression in the program with primitives, that means recusion are not allowed"""
        while flag and itr < BIG_NUM:
            itr += 1
            for derive_name in self.derived:
                if not "{} ".format(derive_name) in flat_string: continue
                formal_params = self.derived[derive_name]["parameters"]
                actual_params, start, end = get_params(flat_string, derive_name)

                """replace derived expressions with the primtives"""
                prefix = flat_string[:start];suffix = flat_string[end:]
                flat_string = "{}{}{}".format(prefix,self.derived[derive_name]["expr"],suffix)

                for i,p in enumerate(formal_params):flat_string = flat_string.replace(p.split("-")[0], actual_params[i])
            
            """until there are no more derived expression in the program"""
            flag = True in [derive in flat_string for derive in self.derived]
        program = Program.parse(flat_string)

        outputs = program.evaluate(context)
        return outputs

    def symbolic_planner(self, start_state, goal_condition):
        pass
    
    def apply_action(self, action_name, params, context):
        """Apply the action with parameters on the given context
        Args:
            action_name: the name of action to apply
            params: a set of integers represent the index of objects in the scene
            context: given all the observable in a diction
        """

        context = copy.copy(context)
        assert action_name in self.actions
        action = self.actions[action_name] # assert the action must be in the action registry

        """Replace the formal parameters in the predcondition into lambda form"""
        formal_params = [p.split("-")[0] for p in action.parameters]
        
        num_objects = context["end"].size(0)

        context_params = {}
        for i,idx in enumerate(params):
            obj_mask = torch.zeros([num_objects])
            obj_mask[idx] = 1.0
            obj_mask = logit(obj_mask)
            context_param = {"context": context}
            context_param["end"] = torch.min(obj_mask, context["end"])
            context_params[i] = context_param#{**context, "end":idx}

        # handle the replacements of precondition and effects
        precond_expr = str(action.precondition)
        for i,formal_param in enumerate(formal_params):precond_expr = precond_expr.replace(formal_param, f"${i}")
        effect_expr = str(action.effect)
        for i,formal_param in enumerate(formal_params): effect_expr = effect_expr.replace(formal_param, f"${i}")

        """Evaluate the probabilitstic precondition (not quantized)"""
        precond = self.evaluate(precond_expr,context_params)["end"].reshape([-1])

        assert precond.shape == torch.Size([1]),print(precond.shape)
        if self.quantized: precond = precond > 0.0 
        else: precond = precond.sigmoid()

        """Evaluate the expressions"""

        effect_output = self.evaluate(effect_expr, context_params)

        for assign in effect_output["end"]:
            condition = torch.min(assign["c"].sigmoid(), precond)
  
            apply_set = assign["x"]
            source_set = assign["y"]
            value = assign["v"]
            apply_to = assign["to"]

            apply_mask = torch.sigmoid(apply_set)
            obj_mask = torch.softmax(source_set, dim = -1)

            while len(value.shape) > len(obj_mask.shape):
                obj_mask = obj_mask[..., None]
                apply_mask = apply_mask[..., None]

            raw_value = torch.sum(obj_mask * value, dim = 0)
  

            if apply_to in effect_output["context"]:
                effect_output["context"][apply_to] = \
                effect_output["context"][apply_to] * (1 - apply_mask) +\
                    raw_value * apply_mask * condition \
                  + effect_output["context"][apply_to] * apply_mask * (1-condition)


        return precond, effect_output["context"]

    def get_implementation(self, func_name):
        func = self.implement_registry[func_name]
        return func

    
    def get_type(self, concept):
        concept = str(concept)
        for key in self.type_constraints:
            if concept in self.type_constraints[key]: return key
        return False
    
    def build_relations(self, scene):
        end = scene["end"]
        features = scene["features"]
        N, D = features.shape
        cat_features = torch.cat([expat(features,0,N),expat(features,1,N)], dim = -1)
        relations = self.relation_encoder(cat_features)
        return relations
    
    def all_embeddings(self):
        return self.concept_vocab, [self.get_concept_embedding(emb) for emb in self.concept_vocab]

    def get_concept_embedding(self,concept):
        concept = str(concept)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        concept_index = self.concept_vocab.index(concept)
        idx = torch.tensor(concept_index).unsqueeze(0).to(device)

        return self.concept_registry(idx)
    
    def search_discrete_state(self, state, goal):
        init_state = QuantizeTensorState(state = state)

        class ActionIterator:
            def __init__(self, actions, state, executor):
                self.actions = actions
                self.action_names = list(actions.keys())
                self.state = state
                self.executor = executor

                self.apply_sequence = []

                num_actions = self.state.state["end"].size(0)
                obj_indices = list(range(num_actions))
                for action_name in self.action_names:
                    params = list(range(len(self.actions[action_name].parameters)))
                    for param_idx in combinations(obj_indices, len(params)):
                        self.apply_sequence.append([
                            action_name, list(param_idx)
                        ])
                random.shuffle(self.apply_sequence)
                self.counter = 0

            def __iter__(self):
                return self
            
            def __next__(self):
                
                if self.counter >= len(self.apply_sequence):raise StopIteration
                context = copy.copy(self.state.state)
                
                action_chosen, params = self.apply_sequence[self.counter]
                #print(action_chosen,params)
                

                precond, state = self.executor.apply_action(action_chosen, params, context = context)
                #print(action_chosen+str(params),context["moving"],"->", state["moving"])
                self.counter += 1
                state["executor"] = None

                return (action_chosen+str(params), QuantizeTensorState(state=state), -1 * torch.log(precond))
        
        if isinstance(goal,str):
            def goal_check(searchState):
                return self.evaluate(goal,
                                 {0:
                                    {"end":searchState.state["end"],
                                    "context":searchState.state}
                                  })["end"] > 0.0
        else:
            goal_check = goal
        
            

        def get_priority(x, y): return 1

        def state_iterator(state: QuantizeTensorState):
            actions = self.actions
            return ActionIterator(actions, state, self)
        
        states, actions, costs, nr_expansions = run_heuristic_search(
            init_state,
            goal_check,
            get_priority,
            state_iterator,
            False,
            100000,
            10
            )
        
        return states, actions, costs, nr_expansions