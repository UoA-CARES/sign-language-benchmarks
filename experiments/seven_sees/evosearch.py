from trainer import Trainer
from random import random
import os

def mutate(weights, strength = 0.05):
    rgb = min(max(weights['rgb'] + (random()*strength) - (random()*strength),0),1)
    skeleton = min(max(weights['skeleton'] + (random()*strength)- (random()*strength),0),1)
    hands = min(max(weights['left_hand'] + (random()*strength) - (random()*strength),0),1)
    
    weights['rgb'] = rgb
    weights['skeleton'] = skeleton
    weights['left_hand'] = hands
    weights['right_hand'] = hands
    return weights

if __name__ == '__main__':
    POPULATION = 8
    MUTATIONSTRENGTH = 0.05
    modality_weights = {'rgb':0.5,
                            'flow': 0, 
                            'depth':0, 
                            'skeleton': 0.5,
                            'face': 0,
                            'left_hand': 0.5,
                            'right_hand': 0.5}
    counter  = 0
    while(1):
        
        

        log = []
        
        bestWeights = None
        bestAccuracy = 0
        for i in range(POPULATION):
            modality_weights_mutated = mutate(modality_weights, MUTATIONSTRENGTH)
            trainer = Trainer(modality_weights=modality_weights_mutated,
                            epochs=1,
                            batch_size=4)
            
            #trainer.train() # Train the model
            result = trainer.validate()

            # Creating a Trainer class changes the current directory
            # to the root of the repository, so change the current directory
            # back to seven_sees if you are working on it again
            os.chdir('./experiments/seven_sees') 
            if(result[0] > bestAccuracy):
                bestAccuracy = result[0]
                bestWeights = modality_weights_mutated

        modality_weights = bestWeights
        log.append([bestAccuracy, modality_weights])

        with open('log.txt', 'a') as f:
            f.write(str(counter) + " " + str(log[-1][0]) + " " + str(log[-1][1]))
            f.write("\n")
        counter+=1