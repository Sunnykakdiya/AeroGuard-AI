from pathlib import Path

# Current working directory
root = Path(".")

# Folders to create

folders = [
    "data",
    "data/raw",
    "data/processed",
    "data/external",

    "notebooks",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/explainability",
    "src/inference",

    "app",
    "app/pages",
    "app/components",

    "artifacts",
    "artifacts/models",
    "artifacts/preprocessors",
    "artifacts/reports",

    "configs",
    "tests",
    "scripts"
]



# Files to create
# files = [
#     "app/main.py",
#     "ingest_sample_kb.py",
#     "requirements.txt",
#     "run.py",
#     ".env",
# ]


# Create folders
for folder in folders:
    (root / folder).mkdir(parents=True, exist_ok=True)


# Create files
# for file in files:
#     (root / file).touch(exist_ok=True)



print("Project structure created successfully.")