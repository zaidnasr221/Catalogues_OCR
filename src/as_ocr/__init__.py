def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'workflow':
        from .workflow import main as workflow_main
        return workflow_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'deliver':
        from .delivery import main as deliver_main
        return deliver_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'present':
        from .presentation import main as present_main
        return present_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'detect':
        from .detection import main as detect_main
        return detect_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'group':
        from .grouping import main as group_main
        return group_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'callouts':
        from .callouts import main as callouts_main
        return callouts_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'components':
        from .components import main as components_main
        return components_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'enrich':
        from .enrichment import main as enrich_main
        return enrich_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        from .exports import main as export_main
        return export_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'pipeline':
        from .pipeline import main as pipeline_main
        return pipeline_main(sys.argv[2:])
    from .indexer import main as index_main
    return index_main()

__all__ = ['main']
