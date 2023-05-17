def extract_single_sequence(line, current_task_token, special_tokens, twin_line=None):
    ordered_special_token_index = sorted([line.index(i) for i in special_tokens if i in line])
    current_task_token_index = line.index(current_task_token)
    pos_of_current_task_token = ordered_special_token_index.index(current_task_token_index)
    if not twin_line:
        output = line[current_task_token_index+1:ordered_special_token_index[pos_of_current_task_token+1]]
    else:
        assert len(line) == len(twin_line)
        output = twin_line[current_task_token_index+1:ordered_special_token_index[pos_of_current_task_token+1]]
    return output

def extract_targeted_sequence(hybrid_text_align_document, task):
    output = ""
    special_tokens = {"<LEX>", "<ALI>", "<TGT>", "<EOS>"}
    current_task_token = "<{}>".format(task.upper())
    if hybrid_text_align_document.startswith("\n"):
        hybrid_text_align_document = " " + hybrid_text_align_document
    hybrid_text_align_document = hybrid_text_align_document[:-1].split("\n")
    for line in hybrid_text_align_document:
        line = line.split(" ")
        if current_task_token not in line:
            output += " " + "\n"
            continue
        line.append("<EOS>")
        tmp = extract_single_sequence(line, current_task_token, special_tokens)
        output += " ".join(tmp) + "\n"
    return output

if __name__ == "__main__":
    import sys
    task = sys.argv[1]
    output = ""
    line = extract_targeted_sequence(sys.stdin.read(), task)
    print(line, end="")