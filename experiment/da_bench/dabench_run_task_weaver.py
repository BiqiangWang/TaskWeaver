import json
import os
from pathlib import Path

from experiment.da_bench.util.DABENCH import DABench
from taskweaver.app import TaskWeaverApp

DABENCH_PATH = Path(Path(__file__).resolve().parent.parent, "da_bench")
RES_PATH = DABENCH_PATH / "data"


def check_file_exist(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)


def initialize_record(record_path):
    check_file_exist(Path(record_path))
    with open(record_path, "r") as f:
        record_data = f.read().splitlines()
    if record_data:
        id_list = json.loads(record_data[0])
        predictions = json.loads(record_data[1])
        token_cost = int(record_data[2])
    else:
        id_list, predictions, token_cost = [], [], 0
    return id_list, predictions, token_cost


def get_task_weaver_session():
    app_dir = "./project/"
    app = TaskWeaverApp(app_dir=app_dir)
    return app.get_session()


async def evaluate_all(model="gpt_4o_mini", agent="task_weaver"):
    bench = DABench()
    record_path = RES_PATH / f"{agent}_{model}_dabench_onprogress_record.md"
    id_list, predictions, token_cost = initialize_record(record_path)
    for key, value in bench.answers.items():
        if key in id_list:
            continue
        session = get_task_weaver_session()
        requirement = bench.generate_formatted_prompt(key)
        response_round = session.send_message(requirement)
        print(response_round.to_dict())
        break


if __name__ == '__main__':
    evaluate_all()
