# Import necessary modules
import logging
import torch
import re

def format_to_wa(cfg, model):
    """
    Formats the input data and performs inference using the given model to create .wa files.

    Args:
        cfg (object): Configuration object containing settings for the inference process.
        model (object): Pre-trained model used for inference.

    Returns:
        None
    """
    # Compile regular expressions for pattern matching
    alignment_start_pattern = re.compile(r"<alignment>")
    alignment_end_pattern = re.compile(r"</alignment>")
    equality_pattern = re.compile(r"<==>")
    doubleslash_pattern = re.compile(r"//")

    # Set up logging
    logger = logging.getLogger("model.wa")
    logger.info("Start inferencing and creating .wa files")
    
    # Initialize variables
    aligment_section = False
    class_dict = {0: 'EQUI', 1 : 'OPPO', 2 : 'REL', 3 : 'SIMI', 4 : 'SPE1', 5 : 'SPE2'}
    processed_file = []
    processed_file.append('<sentence id="1" status="">\n')

    # Construct the output file name
    file_name = cfg.OUTPUT_DIR + "/" + cfg.DATASETS.TEST_WA[14:-4] + "_predicted.wa"

    # Open the input file and process it line by line
    with open(cfg.DATASETS.TEST_WA, 'rb') as fin:
        fin.readline()  # Skip the first line
        for _, line in enumerate(fin):
            processed_line = []
            line = line.decode('latin-1')  # Decode the line from latin-1 encoding
            if alignment_end_pattern.match(line):
                aligment_section = False  # End of alignment section

            if aligment_section:
                # Find positions of special patterns
                equality_pos = [m.start() for m in re.finditer(equality_pattern, line)]
                doubleslash_pos = [m.start() for m in re.finditer(doubleslash_pattern, line)]
                
                # Extract different parts of the line
                alignment = line[:doubleslash_pos[0]]
                value = line[doubleslash_pos[1] + 2 : doubleslash_pos[2]]
                explanation = line[doubleslash_pos[0] + 2 : doubleslash_pos[1]]
                first_chunk = line[doubleslash_pos[2] + 2 : equality_pos[1]]
                second_chunk = line[equality_pos[1] + 4 :]
                
                # Perform inference if the value and explanation are not NIL or NOALI
                if value != " NIL " and explanation != ' ALIC ' and explanation != ' NOALI ':
                    tokens = model.roberta.encode(first_chunk, second_chunk)  # Encode the chunks
                    out1, out2 = model(tokens)  # Perform model inference
                    value = " " + str(round(out1.item())) + " "  # Round and format the output value
                    if explanation.strip() == "EQUI":
                        value = " 5 "  # Special case for EQUI explanation
                    _, exp = torch.max(out2, 1)  # Get the predicted explanation class
                    explanation = " " + class_dict[exp.item()] + " "  # Map the class to its label
                
                # Construct the processed line
                processed_line = alignment + "//" + explanation + "//" + value + "//" + first_chunk + "<==>" + second_chunk
                processed_file.append(processed_line)  # Add to the processed file list
            else:
                processed_file.append(line)  # Add the line as is

            if alignment_start_pattern.match(line):
                aligment_section = True  # Start of alignment section

    # Write the processed lines to the output file
    with open(file_name, "w", newline='', encoding="utf-8") as f:
        for item in processed_file:
            f.write("%s" % item)

    # Log the creation of the output file
    logger.info('Created file %s' % (file_name))
