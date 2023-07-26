from trainer import Trainer

if __name__ == '__main__':
    modality_weights = {'rgb':1,
                        'flow': 1, 
                        'depth':1, 
                        'skeleton': 1,
                        'face': 1,
                        'left_hand': 1,
                        'right_hand': 1}

    trainer = Trainer(modality_weights=modality_weights,
                      epochs=1,
                      batch_size=4)
    
    print(trainer.train())