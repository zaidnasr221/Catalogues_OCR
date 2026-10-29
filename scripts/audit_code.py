"""AST call/import graph with optional observed calls. No source execution/deletion."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tomllib


def audit(root, observed=()):
    root=Path(root).resolve(); definitions={};calls=[];dependencies=[];duplicates=defaultdict(list)
    paths=sorted([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),*root.glob('server/*.py')])
    for path in paths:
        rel=path.relative_to(root)
        module='.'.join((rel.with_suffix('').parts[1:] if rel.parts[0]=='src' else rel.with_suffix('').parts))
        if module.endswith('.__init__'):module=module[:-9]
        tree=ast.parse(path.read_text(encoding='utf-8'));aliases={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names:
                    aliases[alias.asname or alias.name.split('.')[0]]=alias.name
                    dependencies.append(dict(source=module,target=alias.name))
            elif isinstance(node,ast.ImportFrom):
                package=module.rsplit('.',node.level)[0] if node.level else ''
                base='.'.join(p for p in (package,node.module) if p) if node.level else node.module or ''
                for alias in node.names:
                    aliases[alias.asname or alias.name]=base+'.'+alias.name
                    dependencies.append(dict(source=module,target=base+'.'+alias.name))
        def visit(node,scope=()):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                key=module+'.'+'.'.join((*scope,node.name))
                definitions[key]=dict(file=str(rel),line=node.lineno,callers=[],observed=key in observed,
                                      unreachable_statements=[])
                body=ast.Module(body=node.body,type_ignores=[])
                signature=hashlib.sha256(ast.dump(body,include_attributes=False).encode()).hexdigest()
                duplicates[signature].append(key)
                terminated=False
                for statement in node.body:
                    if terminated:definitions[key]['unreachable_statements'].append(statement.lineno)
                    terminated |= isinstance(statement,(ast.Return,ast.Raise))
                scope=(*scope,node.name)
            elif isinstance(node,ast.ClassDef):
                scope=(*scope,node.name)
            if isinstance(node,ast.Call):
                parts=[];value=node.func
                while isinstance(value,ast.Attribute):
                    parts.insert(0,value.attr);value=value.value
                if isinstance(value,ast.Name):parts.insert(0,value.id)
                if parts:
                    target='.'.join([aliases.get(parts[0],module+'.'+parts[0]),*parts[1:]])
                    calls.append(dict(source=module+('.'+'.'.join(scope) if scope else ''),target=target,
                                      file=str(rel),line=node.lineno,resolved=False))
            for child in ast.iter_child_nodes(node):visit(child,scope)
        visit(tree)
    for call in calls:
        if call['target'] in definitions:
            call['resolved']=True;definitions[call['target']]['callers'].append(call['source'])
    entries=tomllib.loads((root/'pyproject.toml').read_text()).get('project',{}).get('scripts',{})
    return dict(entry_points=entries,script_entry_points=[str(p.relative_to(root)) for p in paths if p.parent.name in {'scripts','server'}],
        dependencies=dependencies,calls=calls,definitions=definitions,
        duplicate_bodies=[names for names in duplicates.values() if len(names)>1],
        unused_candidates=[k for k,v in definitions.items() if not v['callers'] and not v['observed']],
        confirmed_unused=[],limitations=['Dynamic calls, imports, callbacks, CLI dispatch and reflection may evade static resolution.',
            'Focused execution evidence shows reachability only; absence does not establish dead code.',
            'No functions are deleted; broad refactoring waits for selected-page validation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execution-evidence',type=Path)
    args=parser.parse_args()
    observed=json.loads(args.execution_evidence.read_text()).get('observed_calls',[]) if args.execution_evidence else []
    result=audit(args.root,set(observed));args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'graph.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Static dependency and call audit','',
        f'{len(result["definitions"])} functions; {sum(c["resolved"] for c in result["calls"])} resolved call sites; '+
        f'{sum(d["observed"] for d in result["definitions"].values())} functions observed in focused tests.','',
        '## CLI entry points','']
    lines += [f'- {name}: {target}' for name,target in result['entry_points'].items()]
    lines += ['', 'CLI subcommand dispatch and script calls are recorded in graph.json.', '', '## Duplicate function bodies','']
    lines += ['- '+', '.join(group) for group in result['duplicate_bodies']] or ['None found.']
    lines += ['', '## Unreachable statements after unconditional return/raise','']
    lines += [f'- {key}: {v["unreachable_statements"]}' for key,v in result['definitions'].items() if v['unreachable_statements']] or ['None found.']
    lines += ['', '## Unused-code conclusions','',
        'No function is confirmed unused from this evidence. Zero-caller candidates are in graph.json for review.',
        'Obsolete logic cannot be established by static absence alone. No deletions or broad refactoring performed.','']
    lines += result['limitations']
    (args.output/'review.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
