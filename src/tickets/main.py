
from .data.runner import runner
from .schemas.tasks import Task

def main()->None:
    runner(Task.ANALYZE)

if __name__ == "__main__":
    main()

