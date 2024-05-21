import logging
import torch
import re

def setup_logger():
    """
    Sets up the logging configuration.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger("model.wa")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def compile_patterns():
    """
    Compiles regular expressions for pattern matching.

    Returns:
        dict: A dictionary containing compiled regular expressions.
    """
    return {
        'alignment_start': re.compile(r"<alignment>"),
        'alignment_end': re.compile(r"</alignment>"),
        'equality': re.compile(r"<==>"),
        'doubleslash': re.compile(r"//")
    }


def process_line(line, patterns, model, class_dict):
    """
    Processes a single line of the input file, performing inference if necessary.

    Args:
        line (str): The line to process.
        patterns (dict): Dictionary of compiled regular expressions.
        model (torch.nn.Module): Pre-trained model used for inference.
        class_dict (dict): Dictionary mapping class indices to class names.

    Returns:
        str: The processed line.
    """
    processed_line = []
    equality_pos = [m.start() for m in re.finditer(patterns['equality'], line)]
    doubleslash_pos = [m.start() for m in re.finditer(patterns['doubleslash'], line)]

    alignment = line[:doubleslash_pos[0]]
    value = line[doubleslash_pos[1] + 2 : doubleslash_pos[2]]
    explanation = line[doubleslash_pos[0] + 2 : doubleslash_pos[1]]
    first_chunk = line[doubleslash_pos[2] + 2 : equality_pos[1]]
    second_chunk = line[equality_pos[1] + 4 :]

    if value != " NIL " and explanation != ' ALIC ' and explanation != ' NOALI ':
        tokens = model.roberta.encode(first_chunk, second_chunk)
        out1, out2 = model(tokens)
        value = " " + str(round(out1.item())) + " "
        if explanation.strip() == "EQUI":
            value = " 5 "
        _, exp = torch.max(out2, 1)
        explanation = " " + class_dict[exp.item()] + " "

    processed_line = alignment + "//" + explanation + "//" + value + "//" + first_chunk + "<==>" + second_chunk
    return processed_line


def format_to_wa(cfg, model):
    """
    Formats the input data and performs inference using the given model to create .wa files.

    Args:
        cfg (object): Configuration object containing settings for the inference process.
        model (object): Pre-trained model used for inference.

    Returns:
        None
    """
    logger = setup_logger()
    patterns = compile_patterns()
    class_dict = {0: 'EQUI', 1: 'OPPO', 2: 'REL', 3: 'SIMI', 4: 'SPE1', 5: 'SPE2'}
    
    logger.info("Start inferencing and creating .wa files")

    alignment_section = False
    processed_file = ['<sentence id="1" status="">\n']
    file_name = cfg.OUTPUT_DIR + "/" + cfg.DATASETS.TEST_WA[14:-4] + "_predicted.wa"

    with open(cfg.DATASETS.TEST_WA, 'rb') as fin:
        fin.readline()
        for _, line in enumerate(fin):
            line = line.decode('latin-1')
            if patterns['alignment_end'].match(line):
                alignment_section = False

            if alignment_section:
                processed_line = process_line(line, patterns, model, class_dict)
                processed_file.append(processed_line)
            else:
                processed_file.append(line)

            if patterns['alignment_start'].match(line):
                alignment_section = True

    with open(file_name, "w", newline='', encoding="utf-8") as f:
        for item in processed_file:
            f.write("%s" % item)

    logger.info('Created file %s' % file_name)
