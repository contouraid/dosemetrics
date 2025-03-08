import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

plt.style.use("dark_background")
figure(figsize=(12, 8), dpi=100)


if __name__ == "__main__":

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_root = "/home/akamath/Documents/data/ICR/output"

    data_struct = pd.read_csv(os.path.join(data_root, "first_last_data.csv"))

    for struct_name in ["Brain", "Brainstem",
                        "PTV", "Chiasm",
                        "Cochlea_L", "Cochlea_R",
                        "OpticNerve_L", "OpticNerve_R",
                        "LacrimalGland_L", "LacrimalGland_R",
                        "Eye_L", "Eye_R"
                        ]:
            overall_compliance = {}
            idx = 0

            for index, row in data_struct.iterrows():
                subject_name = row["case"]
                first_plan = row["first"]
                last_plan = row["last"]

                if (type(first_plan) is not str) or (type(last_plan) is not str):
                    continue
                else:
                    #print(f"Analyzing subject: {subject_name}, ")
                    geometric_differences_file = os.path.join(data_root, subject_name, "geometric_differences.csv")
                    geometric_data = pd.read_csv(geometric_differences_file, index_col=0)

                    a_compliance_file = os.path.join(data_root, subject_name, "a_first_dose_first_structures_compliance.csv")
                    compliance_data = pd.read_csv(a_compliance_file, index_col=0)
                    if struct_name in compliance_data.index:
                        overall_compliance[idx] = [subject_name, compliance_data.loc[struct_name]["Status"],
                                                   compliance_data.loc[struct_name]["Constraint-True"], "a",
                                                   geometric_data.loc[struct_name]["DSC"],
                                                   geometric_data.loc[struct_name]["SurfaceDice"],
                                                   geometric_data.loc[struct_name]["FalseNegative (cc)"],
                                                   geometric_data.loc[struct_name]["HausdorffDistance95 (mm)"],
                                                   geometric_data.loc[struct_name]["SizeChange"]]
                        idx += 1

                    b_compliance_file = os.path.join(data_root, subject_name, "b_first_dose_last_structures_compliance.csv")
                    compliance_data = pd.read_csv(b_compliance_file, index_col=0)
                    if struct_name in compliance_data.index:
                        overall_compliance[idx] = [subject_name, compliance_data.loc[struct_name]["Status"],
                                                   compliance_data.loc[struct_name]["Constraint-True"], "b",
                                                   geometric_data.loc[struct_name]["DSC"],
                                                   geometric_data.loc[struct_name]["SurfaceDice"],
                                                   geometric_data.loc[struct_name]["FalseNegative (cc)"],
                                                   geometric_data.loc[struct_name]["HausdorffDistance95 (mm)"],
                                                   geometric_data.loc[struct_name]["SizeChange"]]
                        idx += 1

                    c_compliance_file = os.path.join(data_root, subject_name, "c_last_dose_last_structures_compliance.csv")
                    compliance_data = pd.read_csv(c_compliance_file, index_col=0)
                    if struct_name in compliance_data.index:
                        overall_compliance[idx] = [subject_name, compliance_data.loc[struct_name]["Status"],
                                                   compliance_data.loc[struct_name]["Constraint-True"], "c",
                                                   geometric_data.loc[struct_name]["DSC"],
                                                   geometric_data.loc[struct_name]["SurfaceDice"],
                                                   geometric_data.loc[struct_name]["FalseNegative (cc)"],
                                                   geometric_data.loc[struct_name]["HausdorffDistance95 (mm)"],
                                                   geometric_data.loc[struct_name]["SizeChange"]]
                        idx += 1

            overall_compliance_df = pd.DataFrame.from_dict(overall_compliance, orient='index', columns=["SubjectName", "Status",
                                                                                                        "Constraint-True", "Situation",
                                                                                                        "DSC",
                                                                                                        "SurfaceDice",
                                                                                                        "VolumeChange",
                                                                                                        "HausdorffDistance95",
                                                                                                        "SizeChange"])
            print("For structure: ", struct_name)
            #print(overall_compliance_df)
            condition_a = overall_compliance_df[overall_compliance_df["Situation"] == "a"]
            condition_b = overall_compliance_df[overall_compliance_df["Situation"] == "b"]
            condition_c = overall_compliance_df[overall_compliance_df["Situation"] == "c"]

            print(f"Number of failures in condition a: {condition_a[condition_a['Status'] == 'Fail'].shape[0]}, out of {condition_a.shape[0]}")
            print(f"Percentage of failures in condition a: {condition_a[condition_a['Status'] == 'Fail'].shape[0] * 100/condition_a.shape[0]:2.3f}")

            print(f"Number of failures in condition b: {condition_b[condition_b['Status'] == 'Fail'].shape[0]}, out of {condition_b.shape[0]}")
            print(f"Percentage of failures in condition b: {condition_b[condition_b['Status'] == 'Fail'].shape[0] * 100/condition_b.shape[0]:2.3f}")

            print(f"Number of failures in condition c: {condition_c[condition_c['Status'] == 'Fail'].shape[0]}, out of {condition_c.shape[0]}")
            print(f"Percentage of failures in condition c: {condition_c[condition_c['Status'] == 'Fail'].shape[0] * 100/condition_c.shape[0]:2.3f}")

            print("---------------------------------------------------")

            plt.figure()
            sns.kdeplot(data=condition_a, x="Constraint-True", color="r", label=f'A, mean:{condition_a["Constraint-True"].mean():.2f}, std:{condition_a["Constraint-True"].std():.2f}')
            sns.kdeplot(data=condition_b, x="Constraint-True", color="g", label=f'B, mean:{condition_b["Constraint-True"].mean():.2f}, std:{condition_b["Constraint-True"].std():.2f}')
            sns.kdeplot(data=condition_c, x="Constraint-True", color="b", label=f'C, mean:{condition_c["Constraint-True"].mean():.2f}, std:{condition_c["Constraint-True"].std():.2f}')
            plt.axvline(x=condition_a["Constraint-True"].mean(), color='r', linestyle='--')
            plt.axvline(x=condition_b["Constraint-True"].mean(), color='g', linestyle='--')
            plt.axvline(x=condition_c["Constraint-True"].mean(), color='b', linestyle='--')
            plt.title(f"Difference of dose to constraint for {struct_name}")
            plt.grid()
            plt.xlim(-60, 60)
            plt.legend()
            plt.show()

            metric = "HausdorffDistance95"
            plt.figure()
            #plt.scatter(condition_a["Constraint-True"], condition_a[metric], marker="x", label='Condition A')
            plt.scatter(condition_b["Constraint-True"], condition_b[metric], marker="o", label='Condition B')
            plt.scatter(condition_c["Constraint-True"], condition_c[metric], marker="*", label='Condition C')
            plt.title(f"{metric} vs Dose difference to constraint for {struct_name}")
            plt.legend()
            plt.xlim(-60, 60)
            plt.grid()
            plt.show()

            plt.figure()
            sns.stripplot(x="Situation", y="Constraint-True", hue="Status", data=overall_compliance_df)
            plt.title(f"Dose difference to constraint for {struct_name}")
            plt.grid()
            plt.show()

            plt.figure()
            sns.stripplot(x="SizeChange", y="Constraint-True", hue="Situation", data=overall_compliance_df)
            plt.title(f"Dose difference to constraint for {struct_name}")
            plt.grid()
            plt.show()

