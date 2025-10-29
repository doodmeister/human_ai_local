import os
import re
from collections import defaultdict

def count_functions_classes(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        classes = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
        functions = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
        async_funcs = len(re.findall(r'^\s*async\s+def\s+\w+', content, re.MULTILINE))
        imports = len(re.findall(r'^\s*(import|from)\s+', content, re.MULTILINE))
        docstrings = content.count('"""') // 2
        type_hints = len(re.findall(r'->\s*[\w\[\],\s]+', content))
        comments = len(re.findall(r'^\s*#', content, re.MULTILINE))
        lines = len(content.split('\n'))
        
        return {
            'classes': classes,
            'functions': functions,
            'async_functions': async_funcs,
            'imports': imports,
            'docstrings': docstrings,
            'type_hints': type_hints,
            'comments': comments,
            'lines': lines
        }
    except:
        return None

src_path = r'C:\dev\human_ai_local\src'
stats = defaultdict(int)
file_count = 0

for root, dirs, files in os.walk(src_path):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.pytest_cache', 'venv']]
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            file_stats = count_functions_classes(filepath)
            if file_stats:
                file_count += 1
                for key, value in file_stats.items():
                    stats[key] += value

print('Code Quality Metrics for src/ directory:')
print('=' * 60)
print(f'Total files analyzed: {file_count}')
print(f'Total lines of code: {stats["lines"]:,}')
print(f'Total classes: {stats["classes"]}')
print(f'Total functions: {stats["functions"]}')
print(f'Total async functions: {stats["async_functions"]}')
print(f'Total imports: {stats["imports"]}')
print(f'Total docstrings: {stats["docstrings"]}')
print(f'Total type hints: {stats["type_hints"]}')
print(f'Total comments: {stats["comments"]}')
print()
print('Code Quality Indicators:')
if stats["functions"] > 0:
    print(f'Documentation ratio: {stats["docstrings"] / stats["functions"] * 100:.1f}% of functions')
print(f'Average lines per file: {stats["lines"] / file_count:.0f}')
print(f'Functions per file: {stats["functions"] / file_count:.1f}')
print(f'Classes per file: {stats["classes"] / file_count:.1f}')
print(f'Comment density: {stats["comments"] / stats["lines"] * 100:.1f}%')
