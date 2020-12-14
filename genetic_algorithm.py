import copy
from nn_architectures import LSTM, BaseLSTM, AttentionLSTM
import numpy as np
import torch

def random_models(num_models, LSTM_type, device):
    models = []
    for _ in range(num_models):
        if LSTM_type == 'base':
            model = LSTM().to(device)
        elif LSTM_type == 'attention':
            model = AttentionLSTM(device = device).to(device)
        else:
            raise Exception("LSTM_type must be 'base' or 'attention'")

        #Not using GD for Genetic Algo, turn off grads
        for param in model.parameters():
            param.requires_grad = False
        
        models.append(model)
    
    return models


def fitness_test(models, eval_func):
    # Score a bunch of models (golf rules)
    scores = np.zeros(len(models))
    for i,model in enumerate(models):
        avg_loss,_ =  eval_func(model = model, epoch = 0, ga = True, prin = False)
        scores[i] = avg_loss
    return scores

def mutate(model, device):
    # Modify weights with Gaussian noise - tunable std dev
    child_model = copy.deepcopy(model)

    mutation_power = 0.1 #tune this

    for param in child_model.parameters():
        param.data += torch.empty(param.shape).normal_(mean=0,std=mutation_power).to(device)

    return child_model.to(device)

def make_babies(models, sorted_inds, device, mutate_best):
    # sorted_inds will be the top group of parents that performed the best
    # !make sure sorted is going the correct direction
    N = len(sorted_inds)

    babies = []
    keep = 1
    # Make babies which are mutations of top parents
    for i in range(len(models)-keep-mutate_best):
        mutate_ind = np.random.choice(sorted_inds)
        babies.append(mutate(models[mutate_ind], device))
    # Mutate the best parent a few times
    for i in range(mutate_best):
        babies.append(mutate(models[sorted_inds[0]], device))
    #Keep the best parent # TODO: implement keep top k
    babies.append(models[sorted_inds[0]])
    best_baby_ind = len(babies)-1 #last bab is superb

    return babies, best_baby_ind

#Training loop!
