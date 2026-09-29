"""Additional role and administrative-context examples for the existing taggers.

Gold spans are created while rendering, not inferred from a name list. People
mentioned as relatives, witnesses or officers are not labeled as crime parties.
All scenarios are synthetic. Screenshot examples belong to regression tests,
not this training corpus.
"""
import json
import random
from datetime import datetime, timedelta
from string import Formatter

from .config import CATEGORIES, SEED
from .robust_corpus import NAMES, PLACES, ITEMS, WEAPONS, perturb

FIELDS = {"suspect": "SUSPECT", "victim": "VICTIM", "location": "LOCATION", "address": "LOCATION",
          "destination": "LOCATION", "weapon": "WEAPON", "item": "PROPERTY", "vehicle": "PROPERTY",
          "cash": "PROPERTY", "item2": "PROPERTY", "item3": "PROPERTY", "date": "DATE", "time": "TIME", "time2": "TIME", "evidence": "EVIDENCE"}

SHORT = {
    "Theft": [
        "{suspect} stole the {item} of {victim}", "{suspect} took {item} from {victim} without permission",
        "{suspect} stole {victim}'s {item}", "the {item} belonging to {victim} was stolen by {suspect}",
        "{victim} was sleeping when {suspect} stole their {item}", "{suspect} has stolen the {item} owned by {victim}",
        "{suspect} quietly took away {victim}'s {item}", "{victim} reported that {suspect} took the {item}",
        "the {item} of {victim} was taken without consent by {suspect}", "{suspect} picked the {item} from {victim}'s pocket",
        "{victim} had their {item} stolen by {suspect}", "{suspect} secretly removed a {item} belonging to {victim}",
    ],
    "Assault": [
        "{suspect} {attack} {victim}", "{victim} was {passive} by {suspect}",
        "{suspect} {attack} {victim} with a {weapon}", "{victim} got {passive} by {suspect}",
        "{suspect} allegedly {attack} {victim}", "{victim} said that {suspect} {attack} them",
        "{suspect} has {passive} {victim}", "{suspect} {attack} {victim} during an argument",
        "{victim} suffered injuries after {suspect} {attack} them", "{suspect} violently {attack} {victim}",
        "{victim} is the person {passive} by {suspect}", "{suspect} caused injuries when they {attack} {victim}",
    ],
    "Robbery": [
        "{suspect} forcibly snatched the {item} of {victim}", "{suspect} robbed {victim} of {item}",
        "{victim} was robbed by {suspect} using a {weapon}", "{suspect} threatened {victim} and took their {item}",
        "{suspect} showed a {weapon} to {victim} and demanded {item}", "{victim} surrendered {item} when threatened by {suspect}",
        "{suspect} pushed {victim} down and snatched their {item}", "{suspect} forced {victim} to give {item}",
        "{victim}'s {item} was snatched with force by {suspect}", "{suspect} grabbed {item} from {victim} at gunpoint",
        "{suspect} took {victim}'s {item} after threatening violence", "{victim} handed over {item} after {suspect} threatened to stab them",
    ],
    "Burglary": ["{suspect} broke into the locked house of {victim} and stole {item}",
                 "{victim}'s locked home was entered by {suspect} through a broken window"],
    "Fraud": ["{suspect} cheated {victim} with a fake investment", "{victim} was tricked into paying money by {suspect}"],
    "Cybercrime": ["{suspect} hacked the account of {victim}", "{victim}'s account was hacked by {suspect}"],
    "Vandalism": ["{suspect} deliberately damaged the {item} of {victim}", "{victim}'s {item} was deliberately smashed by {suspect}"],
    "Arson": ["{suspect} deliberately set fire to the shop of {victim}", "{victim}'s vehicle was intentionally burned by {suspect}"],
}
ATTACKS = [("hit", "hit"), ("killed", "killed"), ("murdered", "murdered"), ("shot", "shot"), ("stabbed", "stabbed"),
           ("beat", "beaten"), ("punched", "punched"), ("slapped", "slapped"), ("attacked", "attacked"), ("assaulted", "assaulted")]


def render_entities(template, values):
    text, spans, pos = [], [], 0
    for literal, key, _, _ in Formatter().parse(template):
        text.append(literal)
        pos += len(literal)
        if key:
            value = str(values[key])
            text.append(value)
            if key in FIELDS:
                spans.append({"start": pos, "end": pos + len(value), "label": FIELDS[key], "text": value})
            pos += len(value)
    return "".join(text), spans


def make_context_report(index, split="train"):
    split_id = ["train", "validation", "development", "test"].index(split)
    rng = random.Random(SEED + 1000003 * (split_id + 1) + index)
    names = rng.sample(NAMES[split], 4)
    # Most new examples use fresh invented names with one to four components.
    # This prevents memorizing the exact names in any user-provided example.
    if rng.random() < .7:
        names = [" ".join("".join(rng.choice(["ra", "vi", "an", "su", "ya", "ma", "jo", "la", "ki", "na", "de", "mar", "sha", "vin", "qo", "lu"])
                                        for _ in range(rng.randint(2, 3))).title() for _ in range(rng.choice([1, 1, 2, 3, 4]))) for _ in range(4)]
    category = CATEGORIES[index % len(CATEGORIES)]
    date = datetime(2023, 1, 1) + timedelta(days=rng.randrange(1600))
    values = dict(zip(["suspect", "victim", "relative", "officer"], names))
    # Keep punctuation inside unfamiliar names, including typographic apostrophes.
    # Use a separate seed so this augmentation doesn't change scenario sampling.
    name_rng = random.Random(SEED + 2000003 * (split_id + 1) + index)
    if index % 5 < 2:
        for key in ("suspect", "victim", "relative", "officer"):
            parts = values[key].split()
            first = parts[0] + "-" + name_rng.choice(["Kai", "Nuru", "Lou", "Min", "Rae", "Jan"])
            last = name_rng.choice(["O’", "D'", "Al-", "O'"]) + parts[-1]
            values[key] = first + " " + last
    values.update(item=rng.choice(ITEMS + ["office bag", "Lenovo gaming laptop", "Dell laptop", "identity cards", "Samsung phone", "passport"]),
                  location=rng.choice(PLACES[split]), destination=rng.choice(PLACES[split]), weapon=rng.choice(WEAPONS),
                  date=date.strftime(rng.choice(["%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"])),
                  time=f"{rng.randrange(24):02}:{rng.randrange(60):02}", time2=f"{rng.randrange(24):02}:{rng.randrange(60):02}",
                  vehicle=rng.choice(["Splendor motorcycle", "Honda scooter", "black motorcycle", "red bicycle"]),
                  cash=f"cash amounting to ₹{rng.randrange(1, 10)},{rng.randrange(100, 999)}/-",
                  age=rng.randrange(20, 70), number=rng.randrange(1, 300), serial=f"R{rng.randrange(1000, 9999)}XYZ",
                  evidence=rng.choice(["written complaint", "CCTV footage", "screenshots", "photographs"]))
    values["address"] = f"House No. {values['number']}, Pocket-{rng.choice('ABC')}, Block-{rng.randrange(1, 9)}, {rng.choice(PLACES[split])}"
    if rng.random() < .5:
        values["location"] = f"{rng.choice(['Central', 'Civic Square', 'Victoria', 'Park Street', 'Grand Avenue'])} {rng.choice(['Metro Station', 'Bus Terminal', 'Market', 'Road'])}, {values['location']}"
    if rng.random() < .6:
        values["destination"] = rng.choice(["Northern Loop", "Eastern Bypass", "Harbour Lane", "West End", "Market Square", "Riverside Road", "Old Bridge"])
    values["attack"], values["passive"] = rng.choice(ATTACKS)
    patterns = SHORT[category]
    # For the rich three categories reserve the final four constructions for
    # validation/development/test; brief categories still share basic grammar.
    variants = range(8) if split == "train" else [8, 9] if split == "validation" else [10] if split == "development" else [11]
    variant = rng.choice(list(variants)) % len(patterns)
    event = patterns[variant]
    form = (index // 8) % 5
    values["item2"] = ["Lenovo LOQ gaming laptop", "Dell laptop", "Samsung mobile phone", "Apple tablet", "HP notebook"][index % 5]
    values["item3"] = ["identity cards", "bank cards", "passport", "keys", "personal documents"][(index // 5) % 5]
    if form < 3:
        template = event
        if rng.random() < .55:
            template += rng.choice([" while they were sleeping", " while he was sleeping", " while she was away", " without any warning", ""])
        if rng.random() < .8:
            template += " on {date}"
        if rng.random() < .35:
            template += " at {time}"
        if rng.random() < .4:
            template += " near {location}"
        if rng.random() < .25:
            template += rng.choice([". The person fled towards {destination}.", ". The accused ran towards the {destination}.",
                                    ". They escaped in the direction of {destination}.", ". Afterwards the offender fled towards the {destination}."])
    elif form == 3:
        # Bystanders and administrative names intentionally have O labels.
        template = rng.choice([
            "Witness {relative} told Officer {officer} that " + event + " on {date} near {location}.",
            event + ". The statement was recorded by Sub-Inspector {officer}. Father name: {relative}.",
            "Complaint about an incident involving {victim}: " + event + ". Investigating officer: {officer}. Case No. {number}.",
        ])
    else:
        if category == "Robbery":
            values["suspect"] = rng.choice(["unknown male youth", "unidentified man", "unknown woman", "two unknown persons"])
            values["item"] = ["office bag", "backpack", "handbag"][index % 3]
            event = ("{suspect} aged about 22-25 years riding a {vehicle} without a number plate came from behind "
                     "and forcibly snatched their {item} containing one black {item2} (Serial No. {serial}), "
                     "important {item3}, and {cash}, causing them to fall and sustain minor injuries")
        header = rng.choice([
            "Information is received today on {date} at {time} hrs via a written complaint submitted by Shri {victim} S/o Late Shri {relative}, aged {age} years, R/o {address} (Mobile: +91-9876543210), stating that ",
            "On {date}, Ms {victim}, D/o {relative}, resident of {address}, submitted a written statement at {time} hrs. The complainant stated that ",
            "Complainant: {victim}\nParent: {relative}\nAddress: {address}\nDate: {date}\nTime: {time}\nNarrative: ",
            "A complaint was lodged by {victim}, residing at {address}. The statement dated {date} at {time} records that ",
        ])
        template = header + "at around {time2} hrs near Gate No. {number}, {location}, " + event
        if category != "Robbery":
            template += ". Property recorded: {item2} (Serial No. {serial}), {item3}, and {cash}"
        template += (". The accused fled towards " + rng.choice(["", "the "]) + "{destination}. "
                     "The statement has been read over and explained to the complainant. "
                     "Under Section 303(2) and 324(4) BNS, the report is registered electronically at the police station. "
                     "The {evidence} is attached to the case file and the investigation is entrusted to Sub-Inspector {officer}.")
    text, spans = render_entities(template, values)
    # Keep most administrative punctuation intact; also teach case-insensitive
    # party roles and source offsets under noisy spacing.
    style = rng.choice([0, 0, 0, 1, 4, 5])
    if form < 3 and rng.random() < .15:
        text = text.title()
        for span in spans:
            span["text"] = text[span["start"]:span["end"]]
    text, spans = perturb(text, spans, rng, style)
    return {"report_id": f"CTX-{split}-{index:06d}", "narrative": text, "crime_type": category,
            "template_group": f"context-{split}-{category}-{variant}-{form}", "split": split,
            "source": "synthetic-role-context-v3.1", "entities_json": json.dumps(spans, ensure_ascii=False)}


def extend_entity_split(base, split, count):
    import pandas as pd
    generated = pd.DataFrame([make_context_report(i, split) for i in range(count)])
    return pd.concat([base, generated], ignore_index=True)
