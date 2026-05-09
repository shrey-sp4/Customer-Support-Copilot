with open(r"c:\Users\shrey\OneDrive\Desktop\Deep_Learning\Project_DL\DL_Project_code\Project_attempt2\support-copilot\src\tools\executor.py", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 135: continue # Skip header and docstring
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent % 4 != 0:
                try:
                    print(f"Line {i+1}: Indent {indent} is not a multiple of 4: {repr(line)}")
                except:
                    print(f"Line {i+1}: Indent {indent} is not a multiple of 4: (contains non-ascii)")
