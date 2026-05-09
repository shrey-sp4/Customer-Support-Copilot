with open(r"c:\Users\shrey\OneDrive\Desktop\Deep_Learning\Project_DL\DL_Project_code\Project_attempt2\support-copilot\src\tools\executor.py", "rb") as f:
    lines = f.readlines()
    for i in range(210, 215):
        print(f"Line {i+1}: {' '.join(f'{b:02x}' for b in lines[i])}")
