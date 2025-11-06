import pandas as pd

filnavn = "eksport sintret.csv"
n = int(input("Hvor mange emner skal findes med tættest modstand? "))

df = pd.read_csv(filnavn, sep=";", encoding="latin1")
df["Modstandsmåling"] = df["Modstandsmåling"].astype(str).str.replace(",", ".").astype(float)
df = df.sort_values(by="Modstandsmåling").reset_index(drop=True)

if len(df) < n:
    print(f"Der er kun {len(df)} rækker i filen — kan ikke finde {n}.")
    exit()

min_diff = float("inf")
start_index = 0
for i in range(len(df) - n + 1):
    diff = df.loc[i + n - 1, "Modstandsmåling"] - df.loc[i, "Modstandsmåling"]
    if diff < min_diff:
        min_diff = diff
        start_index = i

gruppe = df.iloc[start_index:start_index + n][["Emne nummer", "Modstandsmåling"]]
print(f"\nFundet {n} emner med tættest modstand:\n" + "-"*40)
print(gruppe.to_string(index=False))
