import bisect
import itertools
import json
import re

from pathlib import Path

import regex
from tqdm import tqdm


from src import documents, label_config


input_line_re = re.compile(
    r"(?P<left>\d+),(?P<top>\d+),\d+,\d+,(?P<right>\d+),(?P<bottom>\d+),\d+,\d+,(?P<text>.+)"
)


def read_ocr_file(file: Path) -> documents.Document:
    """Reads an OCR file in SROIE format

    Args:
        file (Path): path to the OCR file

    Returns:
        Document: content of the OCR file
    """
    words_seg, positions_seg = [], []
    words, positions = [], []
    with file.open() as file_p:
        for i, line in enumerate(file_p):
            match = input_line_re.match(line)
            assert match, (
                f"Regex matching error at line {i} when reading {file}. "
                f"«{line}» does not match «{input_line_re.pattern}»"
            )
            extract_dict = match.groupdict()

            text: str = extract_dict["text"]
            position = documents.Position(
                *map(
                    int,
                    (
                        extract_dict["left"],
                        extract_dict["top"],
                        extract_dict["right"],
                        extract_dict["bottom"],
                    ),
                )
            )
            words_seg.append(text)
            positions_seg.append(position)
            
    lines: list[tuple[documents.Position, list[int]]] = []
    segments: list[int] = []
    insertion_idx = 0  # to create the first line with the first segment
    for seg_idx, (word_seg, pos_seg) in enumerate(zip(words_seg, positions_seg)):
        for current_line_idx, (pos_line, segments_in_line) in enumerate(lines):
            if pos_seg.bottom < pos_line.top + documents.MIN_OVERLAPPING_LINE * (pos_line.bottom - pos_line.top):
                insertion_idx = 0  #? current_line_idx instead of 0
                break
            else:
                # check if the segment belongs to the current line, i.e. occupies at least X % of the line height
                # and there is no overlap over the x-axis between the line and the segment
                if pos_seg.top <= pos_line.bottom - documents.MIN_OVERLAPPING_LINE * (pos_line.bottom - pos_line.top)\
                        and (pos_seg.right <= pos_line.left or pos_seg.left >= pos_line.right):
                    pos_line.left = min(pos_line.left, pos_seg.left)
                    pos_line.top = min(pos_line.top, pos_seg.top)
                    pos_line.right = max(pos_line.right, pos_seg.right)
                    pos_line.bottom = max(pos_line.bottom, pos_seg.bottom)
                    
                    segments_in_line.append(seg_idx)
                    
                    insertion_idx = -1
                    break
                else:
                    if current_line_idx + 1 < len(lines):  # check if there is a line below the current one
                        if pos_seg.bottom < lines[current_line_idx + 1][0].top + documents.MIN_OVERLAPPING_LINE * \
                                (lines[current_line_idx + 1][0].bottom - lines[current_line_idx + 1][0].top):
                            insertion_idx = current_line_idx + 1
                            break
                    else:
                        insertion_idx = current_line_idx + 1
                        break
        if insertion_idx >= 0:
            lines.insert(insertion_idx, (pos_seg, [seg_idx]))
            insertion_idx = -1

    for _, seg_indices in lines:
        segments.extend(sorted(seg_indices, key=lambda idx: positions_seg[idx].left))

    for seg_idx in segments:
        word_seg, pos_seg = words_seg[seg_idx], positions_seg[seg_idx]
        line_words, line_positions = split_line(word_seg, pos_seg)
        words.extend(line_words)
        positions.extend(line_positions)

    # TODO: better sorting here taking segments into account
    # use this https://github.com/clemsage/unilm/blob/master/layoutlm/examples/seq_labeling/SROIE.py#L133
    # to try to fix it

    return documents.Document(file, words, positions)


def split_line(text: str, position: documents.Position) -> tuple[list[str], list[documents.Position]]:
    """Splits a line of text into words and keep track of the postion of each word

    Args:
        text (str): text to split on a single line
        position (documents.Position): position of the line

    Returns:
        tuple[list[str], list[documents.Position]]: list of words and positions
    """
    words = text.strip().split()
    positions = []

    individual_char_width = (position.right - position.left) / len(
        text
    )  # Assume monospace font
    running_left_pos = position.left
    for word in words:
        running_right_pos = int(
            round(running_left_pos + len(word) * individual_char_width)
        )
        positions.append(
            documents.Position(
                running_left_pos,
                position.top,
                running_right_pos,
                position.bottom,
            )
        )
        running_left_pos = int(round(running_right_pos + individual_char_width))
    positions[-1].right = position.right
    return words, positions


def label_document(document: documents.Document, target_file: Path) -> documents.LabeledDocument:
    """Labels a document given a target file

    Args:
        document (documents.Document): raw document to label
        target_file (Path): target file in a json format

    Returns:
        documents.LabelledDocument: labelled document according to the target
    """
    target = json.load(target_file.open())
    word_starting_position = [0] + list(
        itertools.accumulate(map(lambda s: len(s) + 1, document.words))
    )[:-1]
    whole_text = " ".join(document.words)

    labels = [label_config.ICDARLabel.OTHER] * len(document)

    for label in (label_config.ICDARLabel.COMPANY, label_config.ICDARLabel.ADDRESS, label_config.ICDARLabel.TOTAL, label_config.ICDARLabel.DATE):
        if label.name.lower() not in target:
            continue
        tgt_string = target[label.name.lower()]

        if tgt_string is None:
            continue

        if label != label_config.ICDARLabel.TOTAL:
            # Bisect this ?
            errors = 8
            match = regex.search(
                f"(?:{regex.escape(tgt_string)}){{e<{errors}}}",
                whole_text,
                flags=regex.IGNORECASE | regex.BESTMATCH,
            )
            if match is None:
                errors = 12
                match = regex.search(
                    f"(?:{regex.escape(tgt_string)}){{e<{errors}}}",
                    whole_text,
                    flags=regex.IGNORECASE | regex.BESTMATCH,
            )

        else:
            match = list(
                regex.finditer(
                    f"(?<=TOT[^C]*ROUND[^C]*?)({regex.escape(tgt_string)})",
                    whole_text,
                    flags=regex.IGNORECASE,
                )
            )
            if match is None or len(match) == 0:
                match = list(
                    regex.finditer(
                        f"(?<=TOT[^C]*INCL[^C]*?GST[^C]*)({regex.escape(tgt_string)})(?!.*ROUND)",
                        whole_text,
                        flags=regex.IGNORECASE,
                    )
                )
                if match is None or len(match) == 0:
                    match = list(
                        regex.finditer(
                            f"(?<=TOT[^C]*?)({regex.escape(tgt_string)})",
                            whole_text,
                            flags=regex.IGNORECASE,
                        )
                    )
                    if match is None or len(match) == 0:
                        match = list(
                            regex.finditer(
                                f"\\b({regex.escape(tgt_string)})\\b", whole_text
                            )
                        )
                        if not match:
                            match = list(
                                regex.finditer(
                                    f"({regex.escape(tgt_string)})", whole_text
                                )
                            ) or None

                        if match:
                            match = match[-1]
                    else:
                        match = match[-1]
                else:
                    match = match[0]
            else:
                match = match[0]

        if match is not None:
            span = match.span()
            beg_word_index, end_word_index = retrieve_word_index_from_char_span(
                word_starting_position, span
            )
            # print(word_starting_position, span, beg_word_index, end_word_index)
            for i in range(beg_word_index, end_word_index):
                labels[i] = label

    return documents.LabeledDocument(labels, document)


def retrieve_word_index_from_char_span(
    word_starting_position: list[int], span: tuple[int, int]
) -> tuple[int, int]:
    beg_word_index = bisect.bisect_left(word_starting_position, span[0])
    if beg_word_index >= len(word_starting_position) or word_starting_position[beg_word_index] > span[0]:
        beg_word_index -= 1

    end_word_index = bisect.bisect_left(word_starting_position, span[1])

    return beg_word_index, end_word_index


def get_docs_from_disk(directory: Path) -> list[documents.LabeledDocument]:
    ocr_files = directory / "txt"
    tgt_files = directory / "tgt"

    all_docs = []
    for ocr, tgt in tqdm(zip(ocr_files.glob("*"), tgt_files.glob("*")), total=sum(1 for _ in ocr_files.glob("*"))):
        assert Path(ocr).name == Path(tgt).name
        doc = label_document(read_ocr_file(Path(ocr)), Path(tgt))
        if doc:
            all_docs.append(doc)

    return all_docs
