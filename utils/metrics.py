import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class IntegratedEvaluator:
    def __init__(self, y_true, y_pred, mask, seasonality=1):
        self.y_true = y_true
        self.y_pred = y_pred
        # Broadcast mask to shape (N, T, V)
        self.mask = mask
        self.N, self.T, self.V = y_true.shape
        self.seasonality = seasonality

    def _apply_mask(self, y_true_1d, y_pred_1d, mask_1d):
        valid_idx = mask_1d.astype(bool)
        return y_true_1d[valid_idx], y_pred_1d[valid_idx]

    def _mase(self, y_true_1d, y_pred_1d, mask_1d):
        y_true_masked, y_pred_masked = self._apply_mask(y_true_1d, y_pred_1d, mask_1d)

        mae_model = mean_absolute_error(y_true_masked, y_pred_masked)

        naive_forecast = y_true_1d[:-self.seasonality]
        naive_actuals = y_true_1d[self.seasonality:]
        mask_naive = mask_1d[self.seasonality:] * mask_1d[:-self.seasonality]

        naive_actuals, naive_forecast = self._apply_mask(naive_actuals, naive_forecast, mask_naive)
        if len(naive_actuals) == 0:
            return np.nan

        mae_naive = mean_absolute_error(naive_actuals, naive_forecast)
        return mae_model / mae_naive if mae_naive != 0 else np.nan

    def _monotonicity_violations(self, y_pred_1d, mask_1d):
        mask_bool = mask_1d.astype(bool)
        violations = []
        for i in range(y_pred_1d.shape[0]):
            real_pred = y_pred_1d[i, mask_bool[i]]  # select valid elements
            diff = real_pred[:-1] - real_pred[1:]
            violation_fraction = np.mean(diff > 0)  # vectorized comparison
            violations.append(violation_fraction)
            return np.mean(violations)
    
    def _biomass_violation(self, index_list):
        '''
        Expected index order in the index_list is: [TAGP, TWSO, TMLV, TWST]

        This function calculates a penalty on the deviations from the equality : TAGP = TWSO + TMLV + TWST
        '''
        y_hat = self.y_pred[:, :, index_list]

        total_biomass = torch.tensor(y_hat[:, :, 0])
        partial_biomass = torch.tensor(y_hat[:, :, 1:])
        
        # Compute the biomass penalty
        inconsistence = (total_biomass - partial_biomass.sum(dim=-1)).abs()
        penalty = (inconsistence * self.mask).sum() / (self.mask.sum() + 1e-8)
        return penalty

    def _assimilation_violation(self, index_list):
        '''
        Expected index order in the index_list is: [ASRC, GASS, MRES]

        This function calculates a penalty on the deviations from the equality : ASRC = GASS - MRES
        '''
        y_hat = self.y_pred[:, :, index_list]

        asrc = torch.tensor(y_hat[:, :, 0])
        gass = torch.tensor(y_hat[:, :, 1])
        mres = torch.tensor(y_hat[:, :, 2])
        # Compute the biomass penalty
        inconsistence = (asrc - (gass - mres)).abs()
        penalty = (inconsistence * self.mask).sum() / (self.mask.sum() + 1e-8)
        return penalty
        
    def _dry_matter_increase_violation(self, index_list):
        """
        Expected index order in the index_list is: [TAGP, TWRT, DMI]

        This function calculates a penalty on the deviations from the equality : DMI_{t-1} = (TAGP_{t}-TAGP_{t-1})+(TWRT{t}-TWRT{t-1})
        """
        y_hat = self.y_pred[:, :, index_list]

        prev_y_hat = torch.tensor(y_hat[:, :-1, :])  # Get the previous time step
        curr_y_hat = torch.tensor(y_hat[:, 1:, :])  # Get the current time step

        # Compute the dry matter increase penalty
        tagp_diff = curr_y_hat[:, :, 0] - prev_y_hat[:, :, 0]
        twrt_diff = curr_y_hat[:, :, 1] - prev_y_hat[:, :, 1]
        inconsistence = (prev_y_hat[:, :, 2] - (tagp_diff + twrt_diff)).abs()
        penalty = (inconsistence * self.mask[:,1:]).sum() / (self.mask.sum() + 1e-8)
        return penalty

    def evaluate_per_variable(self):
        results = {"Variable": [], "MAE": [], "RMSE": [], "R2": [], "MASE": [], "MonoViolation": []}

        for v in range(self.V):
            y_true_v = self.y_true[:, :, v]  # (N, T)
            y_pred_v = self.y_pred[:, :, v]  # (N, T)
            mask_v = self.mask    # (N, T)

            # Flattened metrics
            y_true_masked, y_pred_masked = self._apply_mask(y_true_v, y_pred_v, mask_v)
            mae = mean_absolute_error(y_true_masked, y_pred_masked)
            rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
            r2 = r2_score(y_true_masked, y_pred_masked) if len(y_true_masked) > 1 else np.nan
            mase = self._mase(y_true_v, y_pred_v, mask_v)

            # Per-sequence monotonicity violation
            monotonicity_violation = self._monotonicity_violations(y_pred_v, mask_v)

            results["Variable"].append(f"Var_{v}")
            results["MAE"].append(mae)
            results["RMSE"].append(rmse)
            results["R2"].append(r2)
            results["MASE"].append(mase)
            results["MonoViolation"].append(monotonicity_violation)
        return results


    def to_dataframe(self):
        results = self.evaluate_per_variable()
        df = pd.DataFrame(results)
        df.set_index("Variable", inplace=True)
        return df

    def evaluate_last_timestep(self):
        results = {"Variable": [], "MAE": [], "RMSE": [], "R2": []}
        t_last = self.T - 1

        for v in range(self.V):
            y_true_v_last = self.y_true[:, t_last, v]
            y_pred_v_last = self.y_pred[:, t_last, v]
            mask_v_last = self.mask[:, t_last]

            y_true_masked, y_pred_masked = self._apply_mask(y_true_v_last, y_pred_v_last, mask_v_last)
            mae = mean_absolute_error(y_true_masked, y_pred_masked)
            rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
            r2 = r2_score(y_true_masked, y_pred_masked) if len(y_true_masked) > 1 else np.nan

            results["Variable"].append(f"Var_{v}")
            results["MAE"].append(mae)
            results["RMSE"].append(rmse)
            results["R2"].append(r2)

        return pd.DataFrame(results).set_index("Variable")

    def summary(self):
        df = self.to_dataframe()
        return df.mean(numeric_only=True).to_dict()
    

if __name__ == "__main__":
    # write some test code for the IntegratedEvaluator class
    N, T, V = 1000, 200, 5
    y_true = np.random.rand(N, T, V)
    y_pred = np.random.rand(N, T, V)
    mask = np.random.randint(0, 2, (N, T))

    evaluator = IntegratedEvaluator(y_true, y_pred, mask)
    print(evaluator.summary())
    print(evaluator.to_dataframe())
    print(evaluator.evaluate_last_timestep())