"""Independent participant phrasing for the four predeclared dataset splits."""
VICTIM_PATTERNS = {
    "train": [
        "The reporting victim was {victim}.",
        "Victim {victim} provided an account of the event.",
        "The victim was identified as {victim}.",
        "Officers interviewed {victim}, the victim.",
        "According to the complainant, {victim}, the account was accurate.",
        "The person harmed was {victim}, who filed the victim statement.",
        "A statement was received from victim {victim}.",
        "The victim in this report is {victim}.",
        "{victim} was recorded as the victim following an interview.",
        "The complainant was named as {victim}.",
        "Identified as the victim, {victim} requested an update.",
        "The victim, whose identity was confirmed, was {victim}.",
    ],
    "validation": [
        "The statement identifies {victim} as the victim of the event.",
        "Officers confirmed that the complainant was {victim}.",
        "The victim attending the interview gave the name {victim}.",
    ],
    "development": [
        "{victim}, identified in the case as the victim, spoke to investigators.",
        "Investigators took a complaint from the victim named {victim}.",
        "The person listed as the complainant is {victim}.",
    ],
    "test": [
        "Investigators recorded {victim} as the victim in this matter.",
        "The name supplied by the victim for the record was {victim}.",
        "{victim} is the complainant identified in the case record.",
    ],
}
SUSPECT_PATTERNS = {
    "train": [
        "A witness identified {suspect} as the alleged suspect.",
        "The named suspect, {suspect}, was described by a witness.",
        "The alleged offender was named as {suspect}.",
        "Witnesses referred to suspect {suspect}.",
        "The person alleged to be responsible was {suspect}.",
        "{suspect} was described as a possible suspect.",
        "The complaint named {suspect} as the alleged offender.",
        "The suspect in the witness account is {suspect}.",
        "An allegation was made against suspect {suspect}.",
        "The offender was identified by the witness as {suspect}.",
        "The alleged suspect was reported to be {suspect}.",
        "The witness gave {suspect} as the name of the suspect.",
    ],
    "validation": [
        "The alleged suspect identified during the interview was {suspect}.",
        "According to the witness, {suspect} was the suspect.",
        "Officers recorded an allegation concerning suspect {suspect}.",
    ],
    "development": [
        "The case notes describe {suspect} as an alleged offender.",
        "{suspect}, whom the witness called the suspect, was named in the record.",
        "The person named as the suspect in the statement was {suspect}.",
    ],
    "test": [
        "Investigators recorded {suspect} as the alleged suspect in this matter.",
        "The name given for the alleged offender was {suspect}.",
        "{suspect} is the suspect identified in the witness record.",
    ],
}
LOCATION_PATTERNS = [
    "The incident occurred at {location}.",
    "Officers attended the scene at {location}.",
    "The location was recorded as {location}.",
    "The event took place near {location}.",
    "{location} was identified as the scene of the event.",
    "The address description in the report was {location}.",
]
LAYOUTS = [
    ("event", "location", "victim", "suspect"),
    ("victim", "event", "suspect", "location"),
    ("location", "suspect", "event", "victim"),
    ("suspect", "victim", "location", "event"),
    ("event", "victim", "location", "suspect"),
    ("location", "event", "victim", "suspect"),
]
