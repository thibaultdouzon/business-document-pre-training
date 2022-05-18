from dataclasses import dataclass
from pathlib import Path
from typing import Union

from src import label_config

MAGICAL_LINE_HEIGHT_CONSTANT = 15
MIN_OVERLAPPING_LINE = 0.5


@dataclass
class Position:
    """Absolute position of a box in AABB format.
    Top left corner is assumed to be the origin.
    """

    left: Union[int, float]
    top: Union[int, float]
    right: Union[int, float]
    bottom: Union[int, float]

    def normalize(self, width: Union[int, float], height: Union[int, float]):
        self.left = int(1000 * min(1, self.left / width))
        self.top = int(1000 * min(1, self.top / height))
        self.right = int(1000 * min(1, self.right / width))
        self.bottom = int(1000 * min(1, self.bottom / height))
    
    def invert_normalization(self, width: Union[int, float], height: Union[int, float]):
        if width == height == -1:
            return self
        self.left = int(width * self.left / 1000)
        self.top = int(height * self.top / 1000)
        self.right = int(width * self.right / 1000)
        self.bottom = int(height * self.bottom / 1000)
        return self

    def __len__(self):
        return 4

    def __iter__(self):
        yield from (self.left, self.top, self.right, self.bottom)
    
    @property
    def vertical(self):
        return (self.top + self.bottom) / 2
    
    @property
    def horizontal(self):
        return (self.left + self.right) / 2
    
    def __lt__(self, other):
        # First compare vertical position
        if self.vertical < other.vertical - MAGICAL_LINE_HEIGHT_CONSTANT:
            return True
        if self.vertical > other.vertical + MAGICAL_LINE_HEIGHT_CONSTANT:
            return False
        # Then compare horizontal position
        return self.horizontal < other.horizontal

    def export(self):
        return [self.left, self.top, self.right, self.bottom]


@dataclass
class Document:
    """A single document read from the OCR file."""

    filename: Path
    words: list[str]
    positions: list[Position]
    doc_size: tuple[int, int] = (-1, -1)

    def __post_init__(self):
        assert len(self.words) == len(self.positions)
        if self.doc_size == (-1, -1):
            self._width, self._height = self.get_doc_size()
        else:
            self._width, self._height = self.doc_size
        for p in self.positions:
            p.normalize(self.width, self.height)

    def __len__(self):
        return len(self.words)

    def get_doc_size(self):
        max_width = int(max(p.right for p in self.positions) * 1.05)
        max_height = int(max(p.bottom for p in self.positions) * 1.05)
        return max_width, max_height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


@dataclass
class LabeledDocument:
    """A document with a label associated to each word according to the target."""

    labels: list[label_config.TaskLabel]
    document: Document
    part: int = 0

    def __post_init__(self):
        assert len(self.labels) == len(self.document)

    def __iter__(self):
        for i, l in enumerate(self.labels):
            yield self.document.words[i], self.document.positions[i], l

    def __len__(self):
        return len(self.labels)


def load_from_layoutlm_style_dataset(directory: Path, split: str) -> list[LabeledDocument]:
    """[summary]

    Args:
        directory (Path): [description]
        split (str): [description]

    Returns:
        list[document.LabeledDocument]: [description]
    """
    label_file = directory / f"{split}.txt"
    box_file = directory / f"{split}_box.txt"
    image_file = directory / f"{split}_image.txt"

    document_l: list[LabeledDocument] = []
    curr_filename = ""
    
    # Store for blank lines
    blank_line = False

    with (label_file.open("r", encoding="utf8") as lbl_fp,
          box_file.open("r", encoding="utf8") as box_fp,
          image_file.open("r", encoding="utf8") as img_fp):

        for lbl_line, box_line, img_line in zip(lbl_fp, box_fp, img_fp):
            lbl_line, box_line, img_line = map(str.strip, (lbl_line, box_line, img_line))
            if not lbl_line:
                assert box_line == img_line == "", (box_line, img_line)
                blank_line = True
                continue
            
            
            word, lbl = lbl_line.split("\t")
            word_box, box = box_line.split("\t")
            word_img, *_, file = img_line.split("\t")

            file = file.replace(".png", ".txt")

            assert word == word_box == word_img

            # Changing file or encountering a split because document is too long
            if file != curr_filename or blank_line:
                # insert document
                if curr_filename != "":
                    doc_part = document_l[-1].part + 1 if len(document_l) > 0 and document_l[-1].document.filename.name == curr_filename else 0
                    document_l.append(
                        LabeledDocument(
                            labels,
                            Document(
                                Path(curr_filename),
                                words,
                                boxes,
                                (1000, 1000)
                            ),
                            doc_part
                        )
                    )

                words: list[str] = []
                boxes: list[Position] = []
                labels: list[label_config.TaskLabel] = []

                curr_filename = file
                blank_line = False

            words.append(word)
            boxes.append(Position(*map(int, box.split())))
            if lbl == "O":
                labels.append(label_config.TaskLabel.OTHER)
            elif hasattr(label_config.TaskLabel, lbl.split("-")[1].upper()):
                labels.append(getattr(label_config.TaskLabel, lbl.split("-")[1].upper()))
            else:
                labels.append(label_config.TaskLabel.OTHER)
                
                
    doc_part = document_l[-1].part + 1 if len(document_l) > 0 and document_l[-1].document.filename.name == curr_filename else 0
    document_l.append(
        LabeledDocument(
            labels,
            Document(
                Path(curr_filename),
                words,
                boxes,
                (1000, 1000)
            ),
            doc_part
        )
        
    )
    return document_l