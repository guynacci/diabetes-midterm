import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    # pandas reads the data. matplotlib makes graphs. marimo displays text.
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, pd, plt


@app.cell
def _(mo):
    # State the question and identify the project before the analysis begins.
    # GitHub repository: https://github.com/guynacci/diabetes-midterm
    mo.md("""
    # Hospital stays, medications, and readmission
    **CAI5735 Mid-Term Project | October 15, 2026**
    **Name: Guynacci Bonhomme**

    ## Research question
    How does the number of medications relate to the length of a hospital stay,
    and does this relationship differ by readmission status?

    We compare `time_in_hospital`, `num_medications`, and `readmitted`.
    Each row represents a hospital stay, so the results describe stays rather
    than necessarily different patients.

    ## Dataset
    The course-provided Diabetes 130-US file contains 3,000 stays and 10 columns.
    Its source covers 130 US hospitals during 1999-2008. The analysis uses the
    provided simplified file, not the full original dataset.

    Source: course handout, *Diabetes 130-US Hospitals*. Original data:
    https://doi.org/10.24432/C5230J
    """)
    return


@app.cell
def _(pd):
    # Read the CSV. Keep it in the folder where you start marimo.
    df = pd.read_csv("diabetes_130.csv")
    print("Rows and columns:", df.shape)
    # Show only the three columns used in the research question.
    print(df[["time_in_hospital", "num_medications", "readmitted"]].head())
    return (df,)


@app.cell
def _(df):
    # Check for missing values and rows with matching values.
    print(df.isna().sum())
    print("Matching rows:", df.duplicated().sum())
    # Count the stays in each readmission group.
    print(df["readmitted"].value_counts())
    # Check the smallest and largest values.
    print("Stay range:", df["time_in_hospital"].min(), df["time_in_hospital"].max())
    print("Medication range:", df["num_medications"].min(), df["num_medications"].max())
    return


@app.cell
def _(mo):
    # Explain why the analysis keeps all rows rather than removing matching records.
    mo.md("""
    ## Data quality decisions
    There are no missing values. Stay lengths range from 1 to 14 days and
    medication counts range from 1 to 75. All readmission values belong to the
    three expected categories.

    Three rows match earlier rows across all available columns. We keep them
    because the file has no encounter or patient ID to confirm duplicate stays.
    Different stays can have the same values in a simplified dataset.

    Medication count means the number of different medications recorded during
    the stay. It does not measure dosage or the number of doses.
    """)
    return


@app.cell
def _(df):
    # Copy the data before adding a new column.
    analysis_df = df.copy()
    # True means the stay lasted at least 7 days. False means less than 7 days.
    analysis_df["long_stay"] = analysis_df["time_in_hospital"] >= 7
    # Filtering keeps only the rows that meet this condition.
    long_stays = analysis_df[analysis_df["long_stay"] == True]
    print("Stays of at least 7 days:", len(long_stays))
    # Sort from the longest stay to the shortest stay.
    sorted_stays = analysis_df.sort_values("time_in_hospital", ascending=False)
    print(sorted_stays[["time_in_hospital", "num_medications", "readmitted"]].head())
    return (analysis_df,)


@app.cell
def _(analysis_df):
    # Group by readmission status and calculate averages.
    mean_values = analysis_df.groupby("readmitted")[["time_in_hospital", "num_medications"]].mean()
    print("Group averages:")
    print(mean_values.round(2))
    # Calculate the middle value in each group.
    median_values = analysis_df.groupby("readmitted")[["time_in_hospital", "num_medications"]].median()
    print("Group medians:")
    print(median_values)
    # Make three smaller tables so each group is easy to use.
    no_readmission = analysis_df[analysis_df["readmitted"] == "Not readmitted"]
    after_30 = analysis_df[analysis_df["readmitted"] == "After 30 days"]
    within_30 = analysis_df[analysis_df["readmitted"] == "Within 30 days"]
    # Correlation tells us how stay length and medication count move together.
    overall_r = analysis_df["time_in_hospital"].corr(analysis_df["num_medications"])
    no_r = no_readmission["time_in_hospital"].corr(no_readmission["num_medications"])
    after_r = after_30["time_in_hospital"].corr(after_30["num_medications"])
    within_r = within_30["time_in_hospital"].corr(within_30["num_medications"])
    print("Overall correlation:", round(overall_r, 3))
    print("No readmission:", round(no_r, 3))
    print("After 30 days:", round(after_r, 3))
    print("Within 30 days:", round(within_r, 3))
    return after_30, no_readmission, within_30


@app.cell
def _(analysis_df):
    # Compare medication averages using the new True/False column.
    print("Mean medication count:")
    print(analysis_df.groupby(["readmitted", "long_stay"])["num_medications"].mean().round(2))
    print("Number of stays:")
    print(analysis_df.groupby(["readmitted", "long_stay"]).size())
    return


@app.cell
def _(analysis_df, plt):
    # A histogram counts how many stays fall at each length.
    fig_hist = plt.figure(figsize=(8, 4.5))
    plt.hist(analysis_df["time_in_hospital"], bins=14, range=(0.5, 14.5), edgecolor="white")
    plt.title("Distribution of hospital stay length")
    plt.xlabel("Hospital stay (days)")
    plt.ylabel("Number of hospital stays")
    plt.tight_layout()
    fig_hist
    return


@app.cell
def _(after_30, no_readmission, plt, within_30):
    # A boxplot compares the medication counts in the three groups.
    fig_box = plt.figure(figsize=(8, 4.5))
    plt.boxplot([no_readmission["num_medications"], after_30["num_medications"], within_30["num_medications"]])
    plt.xticks([1, 2, 3], ["Not readmitted", "After 30 days", "Within 30 days"])
    plt.title("Medication counts by readmission status")
    plt.xlabel("Readmission status")
    plt.ylabel("Different medications during stay")
    plt.tight_layout()
    fig_box
    return


@app.cell
def _(after_30, no_readmission, plt, within_30):
    # Each dot is one stay. Different colors show the readmission groups.
    # alpha makes the dots see-through because many stays overlap.
    fig_scatter = plt.figure(figsize=(10, 4.5))
    plt.scatter(no_readmission["time_in_hospital"], no_readmission["num_medications"], alpha=0.25, label="Not readmitted")
    plt.scatter(after_30["time_in_hospital"], after_30["num_medications"], alpha=0.25, label="After 30 days")
    plt.scatter(within_30["time_in_hospital"], within_30["num_medications"], alpha=0.25, label="Within 30 days")
    plt.title("Stay length and medication count by readmission status")
    plt.xlabel("Hospital stay (days)")
    plt.ylabel("Different medications during stay")
    plt.legend()
    plt.tight_layout()
    fig_scatter
    return


@app.cell
def _(after_30, no_readmission, plt, within_30):
    # Compare shorter and longer stays within each readmission group.
    # Each list contains three averages in the same group order.
    short_means = [
        no_readmission[no_readmission["long_stay"] == False]["num_medications"].mean(),
        after_30[after_30["long_stay"] == False]["num_medications"].mean(),
        within_30[within_30["long_stay"] == False]["num_medications"].mean()
    ]
    long_means = [
        no_readmission[no_readmission["long_stay"] == True]["num_medications"].mean(),
        after_30[after_30["long_stay"] == True]["num_medications"].mean(),
        within_30[within_30["long_stay"] == True]["num_medications"].mean()
    ]
    # Place the two sets of bars next to each other.
    fig_bar = plt.figure(figsize=(8, 4.5))
    plt.bar([0.8, 1.8, 2.8], short_means, width=0.4, label="1 to 6 days")
    plt.bar([1.2, 2.2, 3.2], long_means, width=0.4, label="7 or more days")
    plt.xticks([1, 2, 3], ["Not readmitted", "After 30 days", "Within 30 days"])
    plt.title("Mean medication count by stay length and readmission")
    plt.xlabel("Readmission status")
    plt.ylabel("Mean different medications during stay")
    plt.ylim(0, 30)
    plt.legend()
    plt.tight_layout()
    fig_bar
    return


@app.cell
def _(mo):
    # Explain the results in plain language.
    mo.md("""
    ## Findings
    Longer stays usually had more medications. The overall correlation was
    **0.463**. The correlations were **0.459** for no readmission, **0.459**
    for readmission after 30 days, and **0.498** for readmission within 30 days.
    The numbers are close, so the pattern looks similar in all three groups.
    I did not test whether the differences were statistically significant.

    Stays followed by readmission within 30 days averaged **5.09 days** and
    **16.45 medications**. Stays with no recorded readmission averaged
    **4.11 days** and **15.15 medications**. The boxplots overlap a lot.
    Medication count alone does not clearly separate the groups.
    In every readmission group, stays of at least seven days had more medications
    on average than shorter stays.

    ## Limitations
    These results do not prove that medications cause longer stays or readmission.
    Sicker patients might stay longer and need more medications.
    Seven days is just a cutoff I chose for the comparison.
    The data are older, and the course file does not explain how it selected
    these 3,000 stays. There are no patient IDs to identify repeat stays.

    ## Answer
    Medication count generally increases with stay length. This pattern looks
    similar across the three readmission groups.

    ## Reference
    Course handout: Diabetes 130-US Hospitals.
    Strack et al. (2014). Impact of HbA1c measurement on hospital readmission rates.
    BioMed Research International.
    Original dataset: https://doi.org/10.24432/C5230J
    """)
    return


if __name__ == "__main__":
    app.run()
