import configparser

from enum import Enum

from src import parser

class Label(int, Enum):
    """Classification label."""


class ICDARLabel(Label):
    """Classification label for ICDAR"""

    OTHER = 0
    COMPANY = 1
    ADDRESS = 2
    TOTAL = 3
    DATE = 4

    @staticmethod
    def export_fields():
        return [
            "COMPANY",
            "ADDRESS",
            "TOTAL",
            "DATE",
        ], []

    @classmethod
    def from_str(cls, s: str):
        if hasattr(cls, s.upper()):
            return getattr(cls, s.upper())
        if s.upper() == "NONE":
            return cls.OTHER
        if s.upper() == "SUPPLIER":
            return cls.COMPANY
        return cls.OTHER

    def get_parser(self):
        if self in (ICDARLabel.TOTAL, ):
            return parser.amount_parser
        elif self in (ICDARLabel.DATE, ):
            return parser.date_parser
        return parser.identity_parser

    
class PO51kLabel(Label):
    """Classification label for PO51k"""

    OTHER = 0
    SALESORDERNUMBER = 1
    SALESORDERDATE = 2
    TOTALPRICE = 3

    @staticmethod
    def export_fields():
        return [
            "SALES_ORDER_NUMBER",
            "SALESORDERDATE",
            "TOTALPRICE",
        ], []

    @classmethod
    def from_str(cls, s: str):
        if hasattr(cls, s.upper()):
            return getattr(cls, s.upper())
        if s.upper() == "NONE":
            return cls.OTHER
        if s.upper() == "SALES_ORDER_NUMBER":
            return cls.SALESORDERNUMBER
        if s.upper() == "DATE":
            return cls.SALESORDERDATE
        if s.upper() == "TOTAL":
            return cls.TOTALPRICE
        return cls.OTHER

    def get_parser(self):
        if self in (PO51kLabel.TOTALPRICE, ):
            return parser.amount_parser
        elif self in (PO51kLabel.SALESORDERDATE, ):
            return parser.date_parser
        return parser.identity_parser


class TaskType(int, Enum):
    SEQUENCE_TAGGING = 0
    SPAN_QUESTION_ANSWERING = 1
    GEN_QUESTION_ANSWERING = 2


conf = configparser.ConfigParser()
conf.read("label.ini")

label_d = {
    "ICDAR": ICDARLabel,
    "PO51k": PO51kLabel,
}

TaskLabel = label_d[conf["LABEL"]["Name"]]
assert hasattr(TaskLabel, "OTHER")

task_d = {
    "SequenceTagging": TaskType.SEQUENCE_TAGGING,
    "SpanQuestionAnswering": TaskType.SPAN_QUESTION_ANSWERING,
    "GenQuestionAnswering": TaskType.GEN_QUESTION_ANSWERING
}
SelectedTask = task_d[conf["TASK"]["Type"]]
