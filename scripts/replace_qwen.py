import os, glob

files = glob.glob('**/*.py', recursive=True) + glob.glob('**/*.md', recursive=True)

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        modified = False
        
        if 'arcee-ai/Arcee-VyLinh' in content or 'arcee-ai/Arcee-VyLinh' in content:
            content = content.replace('arcee-ai/Arcee-VyLinh', 'arcee-ai/Arcee-VyLinh')
            content = content.replace('arcee-ai/Arcee-VyLinh', 'arcee-ai/Arcee-VyLinh')
            modified = True
            
        if modified:
            with open(file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Updated {file}')
    except Exception as e:
        pass
