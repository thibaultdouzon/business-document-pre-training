import re

from dateutil import parser


def identity_parser(s: str):
    return s


amount_re = re.compile(r"\d+(?:[,.]\d+)?")
def amount_parser(s: str):
    s = s.replace(" ", "")
    m = amount_re.findall(s)
    if m:
        val_str = m[0].replace(",", ".")
        return float(val_str)
    return None


class international_parserinfo(parser.parserinfo):
    """
    Class which handles what inputs are accepted. Subclass this to customize
    the language and acceptable values for each parameter.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM
        and YMD. Default is ``False``.

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken
        to be the year, otherwise the last number is taken to be the year.
        Default is ``False``.
    """

    # m from a.m/p.m, t from ISO T separator
    JUMP = [" ", ".", ",", ";", "-", "/", "'",
            "at", "on", "and", "ad", "m", "t", "of",
            "st", "nd", "rd", "th", 
            "le", "a", "à", "er", "eme", "et", "de"]  # FR

    WEEKDAYS = [("Mon", "Monday", "Lundi"),
                ("Tue", "Tuesday", "Mardi"),     # TODO: "Tues"
                ("Wed", "Wednesday", "Mercredi"),
                ("Thu", "Thursday", "Jeudi"),    # TODO: "Thurs"
                ("Fri", "Friday", "Vendredi"),
                ("Sat", "Saturday", "Samedi"),
                ("Sun", "Sunday", "Dimanche")]
    MONTHS = [("Jan", "January", "Jan", "Janvier"),
              ("Feb", "February", "Febr", "Fév", "Fev", "Février", "Fevrier"),
              ("Mar", "March", "Mars"),
              ("Apr", "April", "Avr", "Avril"),
              ("May", "May", "Mai"),
              ("Jun", "June", "Juin"),
              ("Jul", "July", "Juil", "Juillet"),
              ("Aug", "August", "Aout", "Août"),
              ("Sep", "Sept", "September", "Septembre"),
              ("Oct", "October", "Octobre"),
              ("Nov", "November", "Novembre"),
              ("Dec", "December", "Décembre", "Decembre")]
    HMS = [("h", "hour", "hours"),
           ("m", "minute", "minutes"),
           ("s", "second", "seconds")]
    AMPM = [("am", "a"),
            ("pm", "p")]

inter_parserinfo = international_parserinfo(dayfirst=True)

def date_parser(s: str):
    try:
        d = parser.parse(s, inter_parserinfo, fuzzy=True)
        return d.strftime("%Y-%m-%d")
    except:
        return None

