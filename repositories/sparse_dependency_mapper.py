"""
repositories/sparse_dependency_mapper.py

Phase 12B: Sparse Dependency Mapper
Maps code dependencies to sparse retrieval paths, ensuring that when an
agent looks at a file, it also 'retrieves' relevant context from its dependencies.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set

class SparseDependencyMapper:
    """
    Analyzes imports and function calls to build a dependency graph.
    Used to pre-fetch anchors from related files.
    """
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.dependency_graph: Dict[str, Set[str]] = {} # file -> {imported_files}

    def map_dependencies(self):
        """Builds the dependency graph for all python files."""
        for path in self.repo_root.rglob("*.py"):
            if ".git" in str(path): continue
            rel_path = str(path.relative_to(self.repo_root))
            self.dependency_graph[rel_path] = self._extract_imports(path)

    def _extract_imports(self, file_path: Path) -> Set[str]:
        imports = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except Exception:
            pass
        return imports

    def get_contextual_files(self, rel_path: str, depth: int = 1) -> Set[str]:
        """Returns files that are dependencies of the given file."""
        context = set()
        to_visit = [(rel_path, 0)]
        visited = set()

        while to_visit:
            current, current_depth = to_visit.pop(0)
            if current in visited or current_depth > depth:
                continue
            
            visited.add(current)
            if current != rel_path:
                context.add(current)
            
            # Find which files this 'current' file depends on
            # (Requires mapping module names back to file paths)
            deps = self.dependency_graph.get(current, set())
            for dep in deps:
                # Mock resolution: assume module name matches file path
                dep_path = f"{dep.replace('.', '/')}.py"
                if dep_path in self.dependency_graph:
                    to_visit.append((dep_path, current_depth + 1))
        
        return context
