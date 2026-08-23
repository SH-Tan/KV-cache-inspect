import json
import csv
from pathlib import Path
import shutil

problem_dir = Path("./polyglot-benchmark")
output_csv_path = Path("./problem_data.csv")

instructions_addendum = """
####

Use the above instructions to finish the content of this file:
{sol_code}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.
Write {language} code to solve the problem. Present the code in \n```{language}\nYour code\n```\nat the end.
"""

rows = []

for lang_dir in problem_dir.iterdir():
    if not lang_dir.is_dir():
        continue
    
    practice_dir = lang_dir / "exercises" / "practice"
    if not practice_dir.is_dir():
        continue 
    for test_folder in practice_dir.iterdir(): 
        if not test_folder.is_dir():
            continue
        row_dict = {}
        row_dict["folder"] = test_folder.name
        row_dict["language"] = lang_dir.name

        config_path = test_folder / ".meta" / "config.json"
        if not config_path.exists():
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)
            test_file_name = config["files"]["test"][0]
            sol_file_name = config["files"]["solution"][0]

            row_dict["test_file"] = test_file_name
            row_dict["sol_file"] = sol_file_name

            header_content = None

            if lang_dir.name.lower() == "cpp":
                header_files = [f for f in config["files"]["solution"] if f.endswith(".h")]
                if header_files:
                    header_file   = header_files[0]
                    src_header    = test_folder / ".meta" / "example.h"
                    dst_header    = test_folder / header_file

                    if src_header.exists():
                        # copy or overwrite the header into the working dir
                        shutil.copyfile(src_header, dst_header)
                        header_content = src_header.read_text()

            # Load instruction components
            instructions = ""
            intro_path = test_folder / ".docs" / "introduction.md"
            if intro_path.exists():
                instructions += intro_path.read_text()

            instructions += (test_folder / ".docs" / "instructions.md").read_text()

            append_path = test_folder / ".docs" / "instructions.append.md"
            if append_path.exists():
                instructions += append_path.read_text()
            
            if header_content is not None:
                instructions += f"The header file content is: {header_content}"

            # Read solution code
            sol_code_path = test_folder / sol_file_name
            sol_code = sol_code_path.read_text() if sol_code_path.exists() else ""
            instructions += instructions_addendum.format(sol_code=sol_code, language=lang_dir)

            row_dict["instructions"] = instructions

            rows.append(row_dict)

        except Exception as e:
            print(f"⚠️ Failed to process {test_folder.name}: {e}")
            continue

# Save to CSV
with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["folder", "test_file", "sol_file", "instructions", "language"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Saved CSV to {output_csv_path}")