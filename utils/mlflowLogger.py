import mlflow

class LoggerMLflow(object):
    def __init__(self, mlflow_bool: bool):
        self.mlflow_bool = mlflow_bool

    def mlflow_log_metrics(self, metrics: dict, step: int, prefix: str):
            """Logs metrics to MLflow with prefix Train/ or Eval/"""
            if self.mlflow_bool is None:
                return
            namespaced = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
            mlflow.log_metrics(namespaced, step=step)

    def mlflow_log_checkpoint(self, path, artifact_path = "checkpoints"):
        """Uploads checkpoints as MLflow artifacts."""
        if self.mlflow_bool is None:
            return
        mlflow.log_artifact(path, artifact_path)