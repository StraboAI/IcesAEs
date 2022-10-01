from sktime.classification.interval_based import (
    SupervisedTimeSeriesForest,
    RandomIntervalSpectralEnsemble,
    TimeSeriesForestClassifier,
    CanonicalIntervalForest,
)
from sktime.classification.kernel_based import RocketClassifier
from sklearn.ensemble import RandomForestClassifier


class OptunaObjective:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, trial):
        raise NotImplementedError

class RFobjective(OptunaObjective):
    def __call__(self, trial):

        n_estimators = trial.suggest_int("n_estimators", 1, 100, log=True)
        criterion = trial.suggest_categorical("criterion", ['gini',"entropy"])
        max_depth = trial.suggest_int("max_depth", 1, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 2)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        return score

class STSFobjective(OptunaObjective):
    def __call__(self, trial):

        n_estimators = trial.suggest_int("n_estimators", 10, 100, log=True)

        clf = SupervisedTimeSeriesForest(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        return score


class RISFobjective(OptunaObjective):
    def __call__(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)

        clf = RandomIntervalSpectralEnsemble(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        return score

class TSFCobjective(OptunaObjective):
    def __call__(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
        min_interval = trial.suggest_int("min_interval", 3, 10, log=True)

        clf = TimeSeriesForestClassifier(
            n_estimators=n_estimators,
            min_interval=min_interval,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        return score

class CIFobjective(OptunaObjective):
    def __call__(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
        min_interval = trial.suggest_int("min_interval", 3, 10, log=True)
        att_subsample_size = trial.suggest_int("att_subsample_size", 2, 16)

        clf = CanonicalIntervalForest(
            n_estimators=n_estimators,
            min_interval=min_interval,
            att_subsample_size=att_subsample_size,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        return score

class ROCKETobjective(OptunaObjective):
    def __call__(self, trial):
        num_kernels = trial.suggest_int("num_kernels", 1000, 100000, log=True)

        clf = RocketClassifier(
            rocket_transform="minirocket",
            num_kernels=num_kernels,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        return score