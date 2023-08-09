from trainer import Trainer

if __name__ == '__main__':
    rgb_weights = [1.]
    skeleton_weights = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    hand_weights = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    trainer = Trainer(
        epochs=2,
        batch_size=8)

    for rgb_weight in rgb_weights:
        for skeleton_weight in skeleton_weights:
            for hand_weight in hand_weights:

                if skeleton_weight==0.5 and rgb_weight==0.5:
                    continue

                modality_weights = {'rgb':rgb_weight,
                                    'flow': 0, 
                                    'depth':0, 
                                    'skeleton': skeleton_weight,
                                    'face': 0,
                                    'left_hand': hand_weight,
                                    'right_hand': hand_weight}
                
                trainer.set_weights(modality_weights)
    
                trainer.train() # Train the model

                print(trainer.validate())