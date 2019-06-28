import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from datetime import datetime
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

plt.style.use("default")
np.random.seed(88)


def get_cmv_only(df, jason=True):
    """Returns a DataFrame with only CMV specific antigens
    
    Args:
        df (DataFrame): Tidied CMV study DataFrame
        jason (Boolean): Whether to include McLellan pentamer
        
    Returns:
        cmv_df (DataFrame): Tidied DataFrame with only CMV specific antigens
    
    """
    if jason == True:
        cmv_antigens = [
            "prefusion gB",
            "postfusion gB",
            "CG1",
            "CG2",
            "pentamer - McLellan",
            "pentamer",
            "gB",
        ]
    else:
        cmv_antigens = ["CG1", "CG2", "pentamer", "gB"]

    regspec = [x.split("_") for x in df.columns[2:]]
    regspec_df = pd.DataFrame(regspec, columns=["Reagent", "Antigen"])

    cmv_regspec = regspec_df[regspec_df["Antigen"].isin(cmv_antigens)].copy()
    cmv_regspec["cols"] = cmv_regspec["Reagent"] + "_" + cmv_regspec["Antigen"]
    cmv_cols = list(cmv_regspec["cols"])

    df_cmv = df[list(df.columns[:2]) + cmv_cols].copy()

    return df_cmv


def sample_cleaner(sample):
    sample = sample.replace("-", "").lower().strip()
    return sample


def drop_missval_cols(df):
    na_series = df.isna().sum().sort_values(ascending=False)
    na_labels = na_series[na_series > 25].index
    new_df = df.drop(na_labels, axis=1)
    return new_df


def find_final_visits(value):
    if value.endswith("v") or value.endswith("v4"):
        return True
    else:
        return False


def find_longitudinal_samples(df):
    final_visits = list(df[df.Sample.apply(find_final_visits)].Sample)
    first_visits = [x.split("v")[0] for x in final_visits]
    longitudinal = first_visits + final_visits
    return longitudinal, first_visits


def split_groups(df):
    # get longitudinal data
    longitudinal, first_visits = find_longitudinal_samples(df)
    # label groups
    df.loc[df.Cohort == "PP_Erasmus", "Group"] = "Erasmus"
    df.loc[df.Sample.isin(longitudinal), "Group"] = "Longitudinal"
    df.Group = df.Group.fillna("Main")
    # label cohorts
    mapper = {
        "PP": "Primary",
        "NP": "Primary",
        "PL": "Latent",
        "NL": "Latent",
        "PP_Erasmus": "Primary",
    }
    df.Cohort = df.Cohort.map(mapper)
    # split groups
    erasmus = df.loc[df.Group == "Erasmus"].copy()
    df = df.loc[(df.Group == "Main") | (df.Sample.isin(first_visits))].copy()
    return df, erasmus


def get_train_test(df, erasmus):
    X_train = df.copy().drop(["Group", "Sample"], axis=1)
    y_train = X_train.pop("Cohort")
    if type(erasmus) != pd.core.frame.DataFrame:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=88, stratify=y_train
        )
    else:
        X_test = erasmus.copy().drop(["Group", "Sample"], axis=1)
        y_test = X_test.pop("Cohort")
    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test):
    pre_steps = [
        ("logscale", FunctionTransformer(np.log1p, validate=True)),
        ("normalize", StandardScaler()),
    ]
    transform = Pipeline(pre_steps)
    transform.fit(X_train)

    cols = X_train.columns

    X_train = transform.transform(X_train)
    X_test = transform.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=cols)
    X_test = pd.DataFrame(X_test, columns=cols)
    return X_train, X_test


def diagnose(clf, X_train, X_test, y_train, thresh=None):
    trainpreds = clf.predict(X_train)
    testpreds = clf.predict(X_test)
    trainprobs = clf.predict_proba(X_train)
    testprobs = clf.predict_proba(X_test)

    if thresh:
        trainpreds = ["Primary" if x[1] >= thresh else "Latent" for x in trainprobs]
        testpreds = ["Primary" if x[1] >= thresh else "Latent" for x in testprobs]
    else:
        thresh = 0.5

    part_one = pd.DataFrame(
        {"Cohort": y_train, "Preds": trainpreds, "Probs": trainprobs[:, 1]}
    )
    part_two = pd.DataFrame(
        {"Cohort": "Erasmus", "Preds": testpreds, "Probs": testprobs[:, 1]}
    )
    plotthis = pd.concat([part_one, part_two], axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.swarmplot(x="Cohort", y="Probs", hue="Preds", data=plotthis, ax=ax)
    plt.axhline(thresh, linestyle="--", c="red")

    pass


def get_cols(df, erasmus):
    new_cols = [x for x in df.columns if x.startswith("IgM") == False]
    return df[new_cols].copy(), erasmus[new_cols].copy()


def classification(df, erasmus, clf, cols, thresh=None):
    X_train, X_test, y_train, y_test = get_train_test(df, erasmus)
    X_train, X_test = preprocess(X_train, X_test)
    X_train, X_test = X_train[cols].copy(), X_test[cols].copy()
    clf.fit(X_train, y_train)
    diagnose(clf, X_train, X_test, y_train, thresh=thresh)
    pass


def plot_tool_test(sfs_obj, clf, erasmus):
    number_features = []
    feature_list = []
    for key in sfs_obj.get_metric_dict().keys():
        number_features.append(key)
        feature_list.append(list(sfs_obj.get_metric_dict()[key]["feature_names"]))

    accuracies = []
    for l in feature_list:
        X_train, X_test, y_train, y_test = get_train_test(df, erasmus)
        X_train, X_test = preprocess(X_train, X_test)
        X_train = X_train[l].copy()
        X_test = X_test[l].copy()
        clf_temp = clf
        clf_temp.fit(X_train, y_train)
        preds = clf_temp.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
    return number_features, accuracies

def save_params(search_obj, filename):
    prep_dict = sfs_logreg.get_metric_dict()
    for x in prep_dict.keys():
        prep_dict[x]['cv_scores'] = list(prep_dict[x]['cv_scores'])
    now = datetime.now().strftime("%Y%m%d")
    json_metrics = json.dumps(prep_dict)
    f = open("../output/params/{}_{}".format(now, filename),"w")
    f.write(json_metrics)
    f.close()
    pass

def sffs_search(df, erasmus, clf, filename):
    X_train, X_test, y_train, y_test = get_train_test(df, erasmus)
    X_train, X_test = preprocess(X_train, X_test)

    stratkfold = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=20, random_state=88
    ).split(X_train, y_train)
    cv = list(stratkfold)

    sfs = SFS(
        clf,
        k_features="best",
        forward=True,
        floating=True,
        verbose=0,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    sfs.fit(X_train, y_train)

    save_params(sfs, filename)

    return sfs


def plot_subsets(search_obj, test, title=None, filename="params"):
    clf = search_obj.estimator
    now = str(datetime.now().strftime("%Y%m%d"))
    num, acc = plot_tool_test(sfs_logreg, sfs_logreg.estimator, test)

    fig, ax = plt.subplots(figsize=(18,8))
    sns.lineplot(x="variable", y="value", data=plot_df, ci="sd", ax=ax, label="Cross-Validation")
    plt.plot(num, acc, label="Holdout")

    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    if title == None:
        plt.title("SFFS Accuracy vs. Number of Features")
    else:
        plt.title(title)

    fcount = len(plot_df.variable.unique())
    plt.xticks(list(range(1,fcount+1)), rotation=90)
    plt.ylim([-0.05,1.15])
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig("../output/{}_{}".format(now, filename))
    pass


def load_data():
    df = pd.read_csv("../data/processed/190530_merged_cmv_remmel.csv")
    df = df.loc[df.Cohort.isin(["PP_Erasmus", "PP", "NL", "NP", "PL"])]
    df = get_cmv_only(df, jason=True)
    df.Sample = df.Sample.apply(sample_cleaner)
    df = drop_missval_cols(df)
    df, erasmus = split_groups(df)
    df, erasmus = get_cols(df, erasmus)
    return df, erasmus


if __name__ == "__main__":
    df, erasmus = load_data()

    sfs_logreg = sffs_search(df, erasmus, LogisticRegression(solver="lbfgs"), "sfs_logreg_erasmus.json")
    sfs_rf = sffs_search(df, erasmus, RandomForestClassifier(n_estimators=100), "sfs_rf_erasmus.json")
    sfs_logreg_alt = sffs_search(df, None, LogisticRegression(solver="lbfgs"), "sfs_logreg_holdout.json")
    sfs_rf_alt = sffs_search(df, None, RandomForestClassifier(n_estimators=100), "sfs_logreg_holdout.json")

    plot_subsets(
        sfs_rf,
        erasmus,
        title="SFFS Accuracy vs. Number of Features | Random Forest | Erasmus Holdout Set",
        filename="sffs_rf_erasmus.png"
    )
    plot_subsets(
        sfs_logreg,
        erasmus,
        title="SFFS Accuracy vs. Number of Features | Logistic Regression | Erasmus Holdout Set",
        filename="sffs_lr_erasmus.png"
    )
    plot_subsets(
        sfs_rf_alt,
        erasmus,
        title="SFFS Accuracy vs. Number of Features | Random Forest | Subsample Holdout Set",
        filename="sffs_rf_subsample.png"
    )
    plot_subsets(
        sfs_logreg_alt,
        erasmus,
        title="SFFS Accuracy vs. Number of Features | Logistic Regression | Subsample Holdout Set",
        filename="sffs_lr_subsample.png"
    )

