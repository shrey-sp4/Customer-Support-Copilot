with open(r"c:\Users\shrey\OneDrive\Desktop\Deep_Learning\Project_DL\DL_Project_code\Project_attempt2\support-copilot\src\tools\executor.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i in range(210, 216):
        print(f"{i+1}: {repr(lines[i])}")
