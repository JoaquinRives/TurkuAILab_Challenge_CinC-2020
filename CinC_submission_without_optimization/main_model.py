"""
Main Model

@author: Joaquin Rives
"""
import numpy as np


class MainModel:

    def __init__(self, stage_1_1, stage_1_2, stage_1_3, stage_1_4, stage_2 ):
        self.stage_1_1 = stage_1_1  # CNN-LSTM
        self.stage_1_2 = stage_1_2  # residual network
        self.stage_1_3 = stage_1_3
        self.stage_1_4 = stage_1_4
        self.stage_2 = stage_2  # LSTM
        self.n_timesteps = 120  # time steps of the stage 2

    def process_preds_1(self, pred_1_1, pred_1_2, pred_1_3, pred_1_4):

        # concatenate predictions
        preds_1 = np.concatenate([pred_1_1, pred_1_2, pred_1_3, pred_1_4], axis=-1)

        # pad stage 2 inputs 
        input_stage_2 = np.zeros((self.n_timesteps, 36))

        if len(preds_1) > 119:
            preds_1 = preds_1[:119]


        input_stage_2[-len(preds_1):, :] = preds_1
        input_stage_2 = np.expand_dims(input_stage_2, axis=0)

        return input_stage_2

    def predict(self, input_data):
        pred_1_1 = self.stage_1_1.predict(input_data)
        pred_1_2 = self.stage_1_2.predict(input_data)
        pred_1_3 = self.stage_1_3.predict(input_data)
        pred_1_4 = self.stage_1_4.predict(input_data)

        input_2 = self.process_preds_1(pred_1_1, pred_1_2, pred_1_3, pred_1_4)
        final_pred_prob = self.stage_2.predict(input_2)
        final_pred_label = np.where(final_pred_prob > 0.5, 1, 0)

        return final_pred_label, final_pred_prob
