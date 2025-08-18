def save_lines_to_file(file_path, start_line, end_line, file_name):
    # print(file_name)
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.readlines()
    with open(file_path, 'w') as f:
        f.writelines(content[start_line:end_line+1])