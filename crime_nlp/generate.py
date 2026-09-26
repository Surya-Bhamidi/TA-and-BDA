"""Seeded fictional narratives with exact offsets, scenario splits and dirty rows.

Names and locations are invented. Labels are generator ground truth, never features.
Each crime has eight action phrasings and six layouts. Role grammar and names
are independently varied with separate train, validation, development and test pools.
"""
import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta
from string import Formatter

from .config import RAW, ARTIFACTS, SEED, write_json
from .corpus_patterns import VICTIM_PATTERNS, SUSPECT_PATTERNS, LOCATION_PATTERNS, LAYOUTS

FIRST = "Ari Mira Dev Tara Nila Rohan Lena Kiran Soren Isha Ravi Elin Neel Anya Asha Milan Vera Arin Niko Zoya".split()
LAST = "Vale Arden Rowan Sen Mehra Bose Quinn Hale Finch Sethi Noor Wren Shah Lane Ray Kapoor Moss Blake Frost Rao".split()
DISTRICTS = ["Northgate", "Eastbank", "Westhaven", "Southridge", "Old Quarter", "Riverside"]
PLACES = ["Cedar Market", "Maple Arcade", "Willow Station", "Lakeview Plaza", "Orchard Lane", "Juniper Square", "Pine Warehouse", "Birch Avenue", "Elm Court", "Amber Bridge", "Ivy Gardens", "Rosewood Depot"]
WEAPONS = ["knife", "metal bar", "handgun", "wooden stick", "glass bottle"]
OBJECTS = ["mobile phone", "laptop", "wallet", "bicycle", "camera", "watch", "handbag", "delivery parcel"]
ACTIONS = {
    "Burglary": [
        "An intruder forced a window open and entered the locked residence to take a {item}.",
        "The rear door lock was broken and a {item} was removed from an unoccupied apartment.",
        "An offender climbed through a kitchen window and searched the cupboards for valuables.",
        "A shutter was pried open at a closed shop and several items were taken from storage.",
        "The occupant returned to find the front lock damaged and a {item} missing indoors.",
        "An unknown person broke into a secured office and removed a {item} from a desk.",
        "A locked house was entered through a damaged balcony door and household property disappeared.",
        "Someone smashed the window of an empty dwelling, entered it, and carried away a {item}.",
    ],
    "Robbery": [
        "An offender threatened a pedestrian and demanded a {item} before escaping on foot.",
        "The complainant surrendered a {item} after an assailant blocked the path and demanded valuables.",
        "A commuter was held at the roadside while an attacker forcibly took a {item}.",
        "An assailant used threats of violence to take a {item} from a shop visitor.",
        "Two offenders cornered a pedestrian and demanded money with an immediate threat of injury.",
        "An attacker restrained the complainant and forcibly removed a {item} from their bag.",
        "A delivery worker handed over cash when confronted by an offender making violent threats.",
        "An aggressor stopped a passerby, threatened to hurt them, and seized their {item}.",
    ],
    "Theft": [
        "A {item} disappeared from an unattended bench without any confrontation.",
        "A person quietly removed a {item} from an open bag on a crowded bus.",
        "An unlocked {item} was taken from outside a cafe while its owner was indoors.",
        "A visitor noticed a {item} missing after leaving it beside a public counter.",
        "Someone picked a {item} from a coat pocket during a busy community event.",
        "A {item} was carried away from an unattended stall without the owner's consent.",
        "A passenger discovered that property had been removed from an open backpack without force.",
        "An unattended {item} was secretly taken while the owner looked away; no threats were reported.",
    ],
    "Assault": [
        "An argument escalated into a physical attack that left a pedestrian injured.",
        "A person struck the complainant repeatedly during a dispute near the entrance.",
        "An assailant pushed a visitor to the ground and kicked them during a confrontation.",
        "A resident sustained bruises after being punched by another person in a queue.",
        "A dispute over noise ended when one participant attacked another and caused injuries.",
        "The complainant was hit during a street altercation and required medical treatment.",
        "A bystander received cuts after being struck in a violent disagreement.",
        "An individual intentionally punched and injured a passerby following a heated argument.",
    ],
    "Fraud": [
        "A caller impersonated a bank employee and persuaded a resident to transfer money.",
        "A forged invoice caused a business to send payment to an unrelated account.",
        "A seller accepted an advance payment for a {item} and supplied a false receipt.",
        "A resident paid a deposit to a person using forged rental documents.",
        "A fake investment adviser promised guaranteed returns and collected a cash deposit.",
        "A forged signature was used to authorize a payment from a business account.",
        "An impersonator used fabricated documents to obtain a payment for a nonexistent service.",
        "A deceptive seller induced a buyer to send funds for goods that were never delivered.",
    ],
    "Cybercrime": [
        "A malicious email link captured account credentials and enabled unauthorized access.",
        "Ransomware encrypted office files and displayed a demand for digital payment.",
        "A cloned login page harvested a password and the account was accessed remotely.",
        "An intruder exploited a weak password to enter a server and copy private files.",
        "Malware installed through an attachment extracted saved passwords from a computer.",
        "A phishing message directed the recipient to a counterfeit account verification site.",
        "An unauthorized remote session downloaded confidential records from a compromised computer.",
        "A hostile program locked digital files after a user opened a deceptive email attachment.",
    ],
    "Vandalism": [
        "Someone deliberately scratched a parked vehicle and damaged its mirrors.",
        "Fresh paint was sprayed across the public wall and a sign was smashed.",
        "An offender broke the windows of an empty bus shelter without taking property.",
        "A public bench was deliberately damaged and nearby lights were shattered.",
        "A shop sign was defaced with paint and its display board was broken.",
        "Several parked bicycles were intentionally damaged near a community building.",
        "A person deliberately cracked the glass of a noticeboard and tore down its frame.",
        "Public property was defaced with graffiti and damaged, with no items reported stolen.",
    ],
    "Arson": [
        "An offender deliberately ignited waste beside a building and the fire spread to a shutter.",
        "A flammable liquid was poured near a doorway and intentionally set alight.",
        "A parked vehicle was deliberately set on fire after fuel was spilled beside it.",
        "Burning material was placed against a storage shed, causing deliberate fire damage.",
        "A witness observed someone light a pile of cartons against an exterior wall.",
        "A fire was intentionally started in an unoccupied kiosk using an accelerant.",
        "Investigators found evidence of intentional ignition and fuel residue beside the burned structure.",
        "Someone intentionally started a blaze inside a vacant warehouse using combustible material.",
    ],
}

FRAMES = [
    "The incident occurred at {location}. The reporting victim was {victim}. A witness identified {suspect} as the alleged suspect.",
    "At {location}, victim {victim} gave a statement. The named suspect, {suspect}, was described by a witness.",
    "Officers attended {location} and spoke with the victim, {victim}. The alleged offender was named as {suspect}.",
    "According to victim {victim}, the incident took place near {location}. Witnesses referred to suspect {suspect}.",
    "The location was recorded as {location}. Suspect {suspect} was named in the statement of victim {victim}.",
    "Victim {victim} reported the incident from {location}. The complaint named {suspect} as a possible suspect.",
    "A statement from {victim}, the victim, placed the event at {location}. The witness described {suspect} as the suspect.",
    "The victim was identified as {victim} at {location}. The person alleged to be responsible was {suspect}.",
]
ENTITY_FIELDS = {"suspect": "SUSPECT", "suspect2": "SUSPECT", "victim": "VICTIM", "victim2": "VICTIM",
                 "location": "LOCATION", "weapon": "WEAPON", "item": "PROPERTY", "date": "DATE", "time": "TIME", "evidence": "EVIDENCE"}


def render(template, values, start=0):
    """Format once while computing offsets; repeated entities stay aligned."""
    parts, spans, pos = [], [], start
    for literal, key, _, _ in Formatter().parse(template):
        parts.append(literal)
        pos += len(literal)
        if key:
            value = str(values[key])
            parts.append(value)
            if key in ENTITY_FIELDS:
                spans.append({"start": pos, "end": pos + len(value), "label": ENTITY_FIELDS[key], "text": value})
            pos += len(value)
    return "".join(parts), spans


def make_report(index, rng):
    category = list(ACTIONS)[index % len(ACTIONS)]
    variant = (index // len(ACTIONS)) % 8
    split = "train" if variant < 5 else {5: "validation", 6: "development", 7: "test"}[variant]
    style = (index // 64) % len(LAYOUTS)
    # First names are disjoint too: sequence models must use context, not memorization.
    name_pool = {"train": FIRST[:14], "validation": FIRST[14:16], "development": FIRST[16:18], "test": FIRST[18:]}[split]
    suspect, victim, suspect2, victim2 = rng.sample([a + " " + b for a in name_pool for b in LAST], 4)
    district = rng.choice(DISTRICTS)
    values = {"suspect": suspect, "victim": victim, "suspect2": suspect2, "victim2": victim2, "location": f"{rng.choice(PLACES)}, {district}", "item": rng.choice(OBJECTS), "weapon": rng.choice(WEAPONS)}
    date = datetime(2023, 1, 1) + timedelta(days=rng.randrange(1096), minutes=rng.randrange(1440))
    # Narrative has no category label, risk label, template id, or train/test marker.
    parts = {"event": ACTIONS[category][variant], "location": LOCATION_PATTERNS[style],
             "victim": rng.choice(VICTIM_PATTERNS[split]), "suspect": rng.choice(SUSPECT_PATTERNS[split])}
    if rng.random() < .10:
        parts["suspect"] = "The suspect has not been identified."
    elif rng.random() < .08:
        parts["suspect"] += " A second alleged suspect was named as {suspect2}."
    if rng.random() < .10:
        parts["victim"] += " Another victim, {victim2}, also supplied a statement."
    template = " ".join(parts[key] for key in LAYOUTS[style])
    if category in {"Assault", "Robbery"} and rng.random() < .8:
        template += " The witness described a {weapon} carried by the assailant."
    elif category == "Arson":
        values["weapon"] = "petrol can"
        template += " A {weapon} was recovered near the scene."
    elif rng.random() < .25:
        template += " No {weapon} was reported."
    values.update(date=f"{date:%Y-%m-%d}", time=f"{date:%H:%M}", evidence=rng.choice(["camera footage", "written statement", "fingerprint samples", "photographs", "access logs"]))
    template += " The report was recorded on {date} at {time}. Officers collected {evidence} for review."
    if rng.random() < .15:
        template += " The initial note contained an abbrev. and a misspelt descrption; a follow-up statement was requested."
    if rng.random() < .28:
        template += " A follow-up visit documented the condition of the scene. Investigators compared witness accounts and requested further recordings. The reporting person received a reference number and instructions for providing additional evidence. No conclusion about responsibility has been reached."
    text, entities = render(template, values)
    return {"report_id": f"SYN-{index:07d}", "narrative": text, "crime_type": category,
            "reported_at": date.isoformat(timespec="seconds"), "district": district,
            "template_group": f"{category}-{variant}-layout{style}", "split": split,
            "source": "synthetic-v2", "entities_json": json.dumps(entities)}


def generate(count=60000, seed=SEED, path=RAW):
    if count < 64:
        raise ValueError("At least 64 reports are needed to cover all scenario groups.")
    rng = random.Random(seed)
    samples, counts = [], Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i in range(count):
            row = make_report(i, rng)
            counts[row["crime_type"]] += 1
            if i < 1500:
                samples.append(row)
            out.write(json.dumps(row) + "\n")
        dirty_counts = {"duplicates": max(1, count // 40), "null_text": max(1, count // 100), "blank_text": max(1, count // 200), "invalid_date": max(1, count // 200)}
        for kind, n in dirty_counts.items():
            for j in range(n):
                row = dict(samples[j % len(samples)])
                if kind != "duplicates":
                    row["report_id"] = f"DIRTY-{kind}-{j:07d}"
                if kind == "null_text":
                    row["narrative"] = None
                elif kind == "blank_text":
                    row["narrative"] = " \t  \n "
                elif kind == "invalid_date":
                    row["reported_at"] = "invalid-date"
                out.write(json.dumps(row) + "\n")
    manifest = {"source": "synthetic-v2", "seed": seed, "valid_unique_reports": count,
                "raw_rows": count + sum(dirty_counts.values()), "injected_errors": dirty_counts,
                "category_counts": dict(counts), "scenario_groups": 384,
                "diversity": "64 incident phrasings x 6 layouts = 384 composite groups; independently varied participant phrasing, unseen first names, unidentified/multiple suspects and multiple victims.",
                "split_policy": "Per crime: incident variants 0-4 train, 5 validation, 6 development, 7 final test. Participant phrasing and first-name pools are also split. Layout grammar and vocabulary remain shared.",
                "limitations": "Fictional template-generated data; metrics do not establish performance on real crime narratives."}
    write_json(ARTIFACTS / "dataset_manifest.json", manifest)
    print(f"Generated {manifest['raw_rows']:,} rows ({count:,} unique valid reports).", flush=True)
    return manifest
