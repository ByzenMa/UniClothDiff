import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='F-PHAB/')
    parser.add_argument('--output', type=str, default='HandPose/')
    parser.add_argument('--num_prev_frames', type=int, default=3)
    parser.add_argument('--num_next_frames', type=int, default=1)
    args = parser.parse_args()

    objects = ['liquid_soap', 'juice', 'milk', 'salt']
    select_nums = [100, 10]

    script_path = os.path.join(os.path.dirname(__file__), 'data_process.py')
    os.makedirs(args.output, exist_ok=True)

    for obj in objects:
        for select_num in select_nums:
            cmd = [
                sys.executable, script_path,
                '--root', args.root,
                '--output', args.output,
                '--num_prev_frames', str(args.num_prev_frames),
                '--num_next_frames', str(args.num_next_frames),
                '--select_num', str(select_num),
                '--obj', obj,
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
