from qd_common import init_logging
import argparse
from taxonomy import gen_term_list
from taxonomy import gen_noffset

def gen_tsv_from_labeling(input_folder, output_folder):
    fs = glob.glob(op.join(input_folder, '*'))
    labels = set()
    def gen_rows():
        for f in fs:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            yaml_file = op.splitext(f)[0] + '.yaml'
            if not op.isfile(yaml_file):
                logging.info('{} not exist'.format(yaml_file))
            with open(yaml_file, 'r') as fp:
                bb_labels = yaml.loads(fp.read())
            for bb_label in bb_labels:
                labels.add(bb_label['class'])
            with open(f, 'r') as fp:
                encoded_im = base64.b64encode(fp.read())
            yield op.basename(f), json.dumps(bb_labels), encoded_im

    tsv_writer(gen_rows(), op.join(output_folder, 'train.tsv'))
    write_to_file('\n'.join(labels), op.join(output_folder, 'labelmap.txt'))

def process_tsv_main(**kwargs):
    if kwargs['type'] == 'gen_tsv':
        input_folder = kwargs['input']
        output_folder = kwargs['ouput']
        gen_tsv_from_labeling(input_folder, output_folder)
    elif kwargs['type'] == 'gen_term_list':
        tax_folder = kwargs['input']
        term_list = kwargs['output']
        gen_term_list(tax_folder, term_list)
    elif kwargs['type'] == 'gen_noffset':
        tax_input_folder = kwargs['input']
        tax_output_folder = kwargs['output']
        gen_noffset(tax_input_folder, tax_output_folder)

def parse_args():
    parser = argparse.ArgumentParser(description='TSV Management')
    parser.add_argument('-t', '--type', help='what type it is: gen_tsv',
            type=str, required=True)
    parser.add_argument('-i', '--input', help='input',
            type=str, required=False)
    parser.add_argument('-o', '--output', help='output',
            type=str, required=False)
    return parser.parse_args()

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    process_tsv_main(**vars(args))

