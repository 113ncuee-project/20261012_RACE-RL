"""Primary seven-model hierarchy suite; legacy synthetic requires explicit flag."""
import sys

if __name__ == '__main__':
    arguments=sys.argv[1:]
    if '--legacy-synthetic' in arguments:
        arguments.remove('--legacy-synthetic')
        from race_rl.model_suite import main
    else:
        from race_rl.accelerators.cnn_search import main
    raise SystemExit(main(arguments))
