
import os
import javalang

def parse_java_files(java_dir):
    parsed_units = []
    for filename in os.listdir(java_dir):
        if filename.endswith(".java"):
            with open(os.path.join(java_dir, filename), "r") as f:
                code = f.read()
            tree = javalang.parse.parse(code)
            parsed_units.append((filename, tree))
    return parsed_units
