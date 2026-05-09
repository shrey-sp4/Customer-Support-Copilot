with open(r"c:\Users\shrey\OneDrive\Desktop\Deep_Learning\Project_DL\DL_Project_code\Project_attempt2\support-copilot\src\tools\executor.py", "rb") as f:
    content = f.read()
    for i, byte in enumerate(content):
        if byte > 127:
            # Find line number
            line_no = content[:i].count(b'\n') + 1
            print(f"Non-ASCII byte {byte} at index {i} (Line {line_no})")
