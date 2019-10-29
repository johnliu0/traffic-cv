import re

def start_cmd_line():
    while True:
        # prompt user input
        print('--> ', end='');
        cmd = input().strip()
        if len(cmd) == 0:
            continue

        # break user input into tokens
        cmd_tokens = re.split(r'\s+', cmd)

        if cmd_tokens[0] == 'detect':
            if len(cmd_tokens) != 2:
                print('\'detect\' takes in only one argument: image file path')
                continue
            print(cmd_tokens[1])
        else:
            print(f'Unknown command \'{cmd_tokens[0]}\'')


if __name__ == '__main__':
    start_cmd_line();
