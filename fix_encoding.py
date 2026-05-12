import os

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if "OK" in content:
                print(f"Replacing in {path}")
                new_content = content.replace("OK", "OK")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
