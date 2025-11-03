import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class IntegratedEvaluator:
    def __init__(self, y_true, y_pred, mask, seasonality=1):
        self.y_true = y_true
        self.y_pred = y_pred
        # Broadcast mask to shape (N, T, V)
        mask = np.repeat(mask[:, :, np.newaxis], y_true.shape[2], axis=2)
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
    
    def _biomass_violations(self, y_pred, mask_1d):
        mask_bool = mask_1d.astype(bool)
        violations = []
        y_tagp = y_pred[:, :, 2]
        y_twso = y_pred[:, :, 3]
        y_twlv = y_pred[:, :, 4]
        y_twst = y_pred[:, :, 5]
        for i in range(y_tagp.shape[0]):
            real_tagp = y_tagp[i, mask_bool[i]]
            real_twso = y_twso[i, mask_bool[i]]
            real_twlv = y_twlv[i, mask_bool[i]]
            real_twst = y_twst[i, mask_bool[i]]
            diff = abs((real_twso + real_twlv + real_twst) - real_tagp)
            violations.append(diff)
            return np.mean(violations)

    def _assimilation_violations(self, y_pred, mask_1d):
        mask_bool = mask_1d.astype(bool)
        violations = []
        y_asrc = y_pred[:, :, 8]
        y_gass = y_pred[:, :, 9]
        y_mres = y_pred[:, :, 10]
        for i in range(y_asrc.shape[0]):
            real_asrc = y_asrc[i, mask_bool[i]]
            real_gass = y_gass[i, mask_bool[i]]
            real_mres = y_mres[i, mask_bool[i]]  
            diff = (real_gass - real_mres) - real_asrc
            violations.append(diff)
            return np.mean(violations)
        
    def _dry_matter_increase_violations(self, y_pred, mask_1d):
        mask_bool = mask_1d.astype(bool)
        violations = []
        y_tagp = y_pred[:, :, 2]
        y_twrt = y_pred[:, :, 6]
        y_dmi = y_pred[:, :, 7]
        for i in range(y_tagp.shape[0]):
            real_tagp = y_tagp[i, mask_bool[i]]
            real_twrt = y_twrt[i, mask_bool[i]]
            real_dmi = y_dmi[i, mask_bool[i]]
            # I want : (tagp_t - tagp_(t-1)) + (twrt_t - twrt_(t-1)) - dmi_t dans la diff
            dmi_pass = np.roll(real_dmi, 1)
            dmi_pass[0] = 0  # Assuming no intake before the first time step
            diff = abs(abs(real_tagp - np.roll(real_tagp, 1)) + abs(real_twrt - np.roll(real_twrt, 1)) - dmi_pass)
            violations.append(diff)
            return np.mean(violations)

    def evaluate_per_variable(self,
                              entire_bool=True):
        results = {"Variable": [], "MAE": [], "RMSE": [], "R2": [], "MASE": [], "MonoViolation": []}
        if entire_bool:
            results["Biomass"] = []
            results["Assim"] = []
            results["Matter"] = []

        for v in range(self.V):
            y_true_v = self.y_true[:, :, v]  # (N, T)
            y_pred_v = self.y_pred[:, :, v]  # (N, T)
            mask_v   = self.mask[:, :, v]    # (N, T)

            # Flattened metrics
            y_true_masked, y_pred_masked = self._apply_mask(y_true_v, y_pred_v, mask_v)
            mae = mean_absolute_error(y_true_masked, y_pred_masked)
            rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
            r2 = r2_score(y_true_masked, y_pred_masked) if len(y_true_masked) > 1 else np.nan
            mase = self._mase(y_true_v, y_pred_v, mask_v)

            # Per-sequence monotonicity violation
            monotonicity_violation = self._monotonicity_violations(y_pred_v, mask_v)
            if entire_bool:
                biomass_violation = self._biomass_violations(self.y_pred, mask_v)
                assimilation_violation = self._assimilation_violations(self.y_pred, mask_v)
                dry_matter_violation = self._dry_matter_increase_violations(self.y_pred, mask_v)

            results["Variable"].append(f"Var_{v}")
            results["MAE"].append(mae)
            results["RMSE"].append(rmse)
            results["R2"].append(r2)
            results["MASE"].append(mase)
            results["MonoViolation"].append(monotonicity_violation)
            if entire_bool: 
                results["Biomass"].append(biomass_violation)
                results["Assim"].append(assimilation_violation)
                results["Matter"].append(dry_matter_violation)

        return results


    def to_dataframe(self,
                     entire_bool=True):
        results = self.evaluate_per_variable(entire_bool=entire_bool)
        df = pd.DataFrame(results)
        df.set_index("Variable", inplace=True)
        return df

    def evaluate_last_timestep(self):
        results = {"Variable": [], "MAE": [], "RMSE": [], "R2": []}
        t_last = self.T - 1

        for v in range(self.V):
            y_true_v_last = self.y_true[:, t_last, v]
            y_pred_v_last = self.y_pred[:, t_last, v]
            mask_v_last = self.mask[:, t_last, v]

            y_true_masked, y_pred_masked = self._apply_mask(y_true_v_last, y_pred_v_last, mask_v_last)
            mae = mean_absolute_error(y_true_masked, y_pred_masked)
            rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
            r2 = r2_score(y_true_masked, y_pred_masked) if len(y_true_masked) > 1 else np.nan

            results["Variable"].append(f"Var_{v}")
            results["MAE"].append(mae)
            results["RMSE"].append(rmse)
            results["R2"].append(r2)

        return pd.DataFrame(results).set_index("Variable")

    def summary(self,
                entire_bool=True):
        df = self.to_dataframe(entire_bool=entire_bool)
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