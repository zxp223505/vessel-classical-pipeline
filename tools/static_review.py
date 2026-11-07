# -*- coding: utf-8 -*-
"""静态审查：未使用的导入、从未被调用的函数、可疑写法（bare except / 可变默认值 / TODO）。"""
import ast
import os
import re

SRC = r"C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\vessel-course\src"
TESTS = r"C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\vessel-course\tests"

files = [os.path.join(SRC, f) for f in sorted(os.listdir(SRC)) if f.endswith(".py")]
files += [os.path.join(TESTS, f) for f in sorted(os.listdir(TESTS)) if f.endswith(".py")]

sources = {f: open(f, encoding="utf-8").read() for f in files}
all_code = "\n".join(sources.values())

print("=== 1. 未使用的导入 ===")
for f, s in sources.items():
    tree = ast.parse(s)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name.split(".")[0])] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    imported[(a.asname or a.name)] = node.lineno
    body = "\n".join(
        l for i, l in enumerate(s.splitlines(), 1)
        if not re.match(r"\s*(from|import)\s", l)
    )
    unused = [n for n in imported if not re.search(rf"\b{re.escape(n)}\b", body)]
    if unused:
        print(f"  {os.path.basename(f)}: {unused}")

print("=== 2. 从未被调用的函数（排除 main/测试） ===")
defined = {}
for f, s in sources.items():
    for node in ast.walk(ast.parse(s)):
        if isinstance(node, ast.FunctionDef):
            defined.setdefault(node.name, []).append(os.path.basename(f))
for name, where in sorted(defined.items()):
    if name in ("main",) or name.startswith("test_") or name == "check":
        continue
    uses = len(re.findall(rf"\b{re.escape(name)}\s*\(", all_code))
    # 定义本身算一次调用计数
    if uses <= 1:
        print(f"  {name} (定义于 {where})：仅定义未调用")

print("=== 3. 可疑写法 ===")
for f, s in sources.items():
    for i, line in enumerate(s.splitlines(), 1):
        if re.search(r"except\s*:", line) or "TODO" in line or "FIXME" in line:
            print(f"  {os.path.basename(f)}:{i}: {line.strip()}")
    tree = ast.parse(s)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            args = getattr(node, "args", None)
            if not args:
                continue
            defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
            for d in defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    print(f"  {os.path.basename(f)}: 可变默认参数 at line {node.lineno}")

print("=== 4. 每个文件的函数/类数量 ===")
for f, s in sources.items():
    tree = ast.parse(s)
    nf = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
    print(f"  {os.path.basename(f):18s} 函数 {nf:2d}")
