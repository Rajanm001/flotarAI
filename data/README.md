# Data

`Assessment_TwitterDataset.csv` is not committed to git (per the assessment
instructions: no large datasets in the repo). It is the file provided by
Floter AI in the assessment ZIP.

To run this project, place `Assessment_TwitterDataset.csv` in this directory.
Expected columns: `UserID, Name, Gender, DOB, Interests, City, Country`.

Run `python scripts/prepare_data.py` to generate the cleaned/split dataset
into `data/processed/` (also gitignored, regenerated on demand).
