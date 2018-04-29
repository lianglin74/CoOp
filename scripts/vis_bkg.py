import uuid
from qd_common import load_from_yaml_file
from qd_common import write_to_yaml_file
import os.path as op
import time
import logging
from qd_common import ensure_directory
from process_tsv import build_taxonomy_impl
from qd_common import init_logging

def get_vis_bkg_folder():
    return op.join(op.dirname(op.dirname(op.realpath(__file__))), 'output', 'vis_bkg')

def get_task_list_file():
    task_list_file = op.join(get_vis_bkg_folder(), 'task_list.yaml')
    return task_list_file

def save_uploaded_file(f, fname):
    from qd_common import ensure_directory
    ensure_directory(op.dirname(fname))
    with open(fname, 'w') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def push_task(request):
    '''
    need to switch to the Queue system
    '''
    out_data = request.POST.get('out_data')
    task_id = str(uuid.uuid4())
    yaml_file = op.join(get_vis_bkg_folder(), task_id, 
            'taxonomy',
            'root.yaml')
    save_uploaded_file(request.FILES['file_input'], yaml_file)
    param = {'task_id': task_id,
            'sub_tasks': []}
    sub_task = {'type': 'taxonomy_to_tsv', 
            'input': op.dirname(yaml_file),
            'data': out_data,
            'datas': [s.strip() for s in
                    request.POST.get('str_datas').split(',')]}
    param['sub_tasks'].append(sub_task)
    task_list_file = get_task_list_file()
    task_list = load_from_yaml_file(task_list_file) if op.isfile(task_list_file) else []
    task_list.append(param)
    write_to_yaml_file(task_list, task_list_file)

    return param

def pop_task():
    task_list_file = get_task_list_file()
    task_list = load_from_yaml_file(task_list_file) if op.isfile(task_list_file) else []
    if len(task_list) > 0:
        result = task_list[0]
        task_list.remove(result)
        write_to_yaml_file(task_list, task_list_file)
        return result
    else:
        return None

def init_logging2(log_path):
    ensure_directory(op.dirname(log_path))
    format_str = '%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s'
    logFormatter = logging.Formatter(format_str)
    rootLogger = logging.getLogger()
    rootLogger.handlers = []
    
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

def execute_task(task):
    init_logging2(op.join(get_vis_bkg_folder(), 
        task['task_id'], 'log.txt'))
    #init_logging()
    for sub_task in task['sub_tasks']:
        if sub_task['type'] == 'taxonomy_to_tsv':
            build_taxonomy_impl(sub_task['input'],
                    datas=sub_task['datas'],
                    data=sub_task['data'])

def main_entry():
    while True:
        task = pop_task()
        if task is None:
            time.sleep(5)
            continue
        execute_task(task)

if __name__ == '__main__':
    main_entry()
