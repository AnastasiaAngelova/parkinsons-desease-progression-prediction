import numpy as np
import lightgbm as lgb

class LightGBMClassifier:
    """LightGBM classifier for multi-class prediction."""
    
    def __init__(self, params, features):
        self.params = params
        self.features = features
        self.model = None
    
    def fit(self, train_df):
        """Train LightGBM model."""
        if self.features is None:
            self.features = [col for col in train_df.columns if col.startswith("v_")]
        
        lgb_train = lgb.Dataset(train_df[self.features], train_df["target"])
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        
        self.model = lgb.train(
            params, 
            lgb_train, 
            num_boost_round=self.params["n_estimators"]
        )
        return self

    def predict_proba(self, test_df):
        """Get prediction probabilities."""
        return self.model.predict(test_df[self.features])

    def predict(self, test_df):
        """Make predictions using optimal SMAPE strategy."""
        probabilities = self.predict_proba(test_df)
        return self._optimize_smape_predictions(probabilities)
    
    def _optimize_smape_predictions(self, probabilities):
        """Optimize predictions for SMAPE metric."""
        def single_smape(preds, target):
            x = np.tile(np.arange(preds.shape[1]), (preds.shape[0], 1))
            x = np.abs(x - target) / (2 + x + target)
            return (x * preds).sum(axis=1)
        
        smape_scores = np.hstack([
            single_smape(probabilities, i).reshape(-1, 1) 
            for i in range(probabilities.shape[1])
        ])
        return smape_scores.argmin(axis=1)