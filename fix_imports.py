import os, re
targets = ["model", "curriculum_dataset", "dataset", "logger", "score_difficulty", "tokenizer", "curriculum", "legal_dataset"]
folders_to_check = [".", "scripts", "tests", "hf_deployment"]
pattern = re.compile(r"^(from |import )(" + "|".join(targets) + r")(\b)", flags=re.MULTILINE)
for folder in folders_to_check:
    if not os.path.exists(folder): continue
    for f in os.listdir(folder):
        if not f.endswith(".py"): continue
        path = os.path.join(folder, f)
        with open(path, "r") as file: content = file.read()
        new_content, count = pattern.subn(r"\g<1>src.\g<2>\g<3>", content)
        if count > 0:
            with open(path, "w") as file: file.write(new_content)
            print(f"Updated {count} imports in {path}")
print("Done")
