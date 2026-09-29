"""Fictional informal English, generated with exact source spans.

This extends the existing template generator, not the inference algorithm.
Names, places, event phrasings and typo seeds are split before fitting. Roles
come from the described action, never from a country's or person's identity.
"""
import json
import random
import re
from datetime import datetime, timedelta

from .config import CATEGORIES, SEED

# Full strings deliberately include mononyms, initials, diacritics, apostrophes,
# hyphens and several naming conventions. All scenarios are invented.
NAMES = {
    "train": ["Aarav", "Priya Nair", "Mohammed Ali", "Fatima Hassan", "Wei Zhang", "Mei Lin",
              "José García", "María del Río", "Jean-Luc Martin", "Anne O'Neill", "Chidi Okafor",
              "Amina Diallo", "Yuki Tanaka", "Soo Min Kim", "Olga Ivanova", "Omar Saleh",
              "R. K. Menon", "Sanjay Kumar", "Lakshmi", "Kwame Mensah", "Alex Morgan",
              "Noor Al-Hassan", "Carlos Santos", "Nguyen Thi Lan", "Siti Nur", "Müller",
              "Abebe Bekele", "Sam Taylor", "Dara", "Nandini Rao", "Ibrahim", "Chen Li"],
    "validation": ["Zeynep Yılmaz", "Kofi Boateng", "Ananya Iyer", "Luis Fernández", "Hana Suzuki",
                   "Siobhán Murphy", "A. Q. Rahman", "Nkosana", "Minh Tran", "Dewi Putri"],
    "development": ["Björk Jónsdóttir", "Adebayo Bello", "Qian Wu", "Renée Dubois", "Aditya Pillai",
                    "F. J. D'Souza", "Mateo Rivera", "Zahra", "Thabo Dlamini", "Nur Aisyah"],
    "test": ["Sławomir Kowalski", "Ximena Núñez", "Tenzin Dorje", "Özlem Demir", "J. P. Dlamini",
             "Aoife Ní Bhraonáin", "Chiamaka Eze", "Pranavi Reddy", "Abdul-Rahim", "Sokha Chan"],
}
PLACES = {
    "train": ["Mumbai", "Delhi", "Nairobi", "Lagos", "London", "São Paulo", "Tokyo", "Cairo",
              "Dubai", "New York", "rural bus stand", "Sector 12", "2nd Cross Road", "Central Market",
              "Rue des Lilas", "Jalan Merdeka", "village school", "Platform 4", "the local market"],
    "validation": ["Kochi", "Kampala", "Seoul", "München", "Oak Street", "Block 19", "the railway station"],
    "development": ["Hyderabad", "Accra", "Jakarta", "Zürich", "Avenida Norte", "Gate 7", "the shopping mall"],
    "test": ["Visakhapatnam", "Ouagadougou", "Reykjavík", "Bogotá", "Ulaanbaatar", "Lane 23", "the ferry terminal"],
}
ITEMS = ["phone", "mobile", "cellphone", "mobile phone", "laptop", "wallet", "purse", "bag", "handbag",
         "gold chain", "jewellery", "jewelry", "bicycle", "bike", "scooter", "cash", "money", "watch", "parcel"]
WEAPONS = ["knife", "gun", "pistol", "iron rod", "stick", "bottle", "wooden stick", "metal bar"]

# Variants 0..11 train, 12..13 validation, 14..15 development, 16..17 test.
# These include short inputs, Indian English word order and common international
# vocabulary; informal English is not equated with any nationality.
EVENTS = {
    "Theft": [
        "my {item} stolen in bus", "someone took my {item} from pocket and went away",
        "{suspect} quietly stole {victim}'s {item}", "{victim} ka {item} someone took without asking",
        "i kept {item} on table now gone someone taken it", "{suspect} picked {item} from {victim}'s bag",
        "my {item} is missing after travelling in crowded train", "{victim} says {item} got stolen nobody threatened",
        "somebody stealing my {item} when i not looking", "{suspect} took away my {item} without permission",
        "{victim}'s {item} was stolen by {suspect}", "someone has taken {item} from my open bag no fight happened",
        "while waiting for bus somebody removed my {item}", "{suspect} secretly carried away {victim}'s {item}",
        "my {item} gone from pocket after crowd pushed me", "{victim} found their {item} stolen by {suspect}",
        "{item} kept beside me disappeared while i was distracted", "{suspect} slipped {victim}'s {item} out of the bag",
    ],
    "Robbery": [
        "two fellows showed {weapon} and took my {item}", "{suspect} threatened {victim} with {weapon} and snatched {item}",
        "he put {weapon} on me asking give {item} or he will hurt me", "my {item} snatched by force after he hit me",
        "{suspect} stopped {victim} and demanded {item} at gunpoint", "they beat me and grabbed my {item}",
        "{victim} was robbed by {suspect} who threatened with {weapon}", "one guy pulled {item} from my hand and pushed me down",
        "{suspect} holding {weapon} forced {victim} to hand over {item}", "someone mugged me and ran with my {item}",
        "{suspect} snatched {victim}'s {item} using force", "{victim} handed {item} to {suspect} after threats to kill",
        "they cornered me with {weapon} and made me give {item}", "{suspect} grabbed {victim}'s {item} after threatening to stab",
        "i got beaten when refusing to give my {item} they took it", "{victim} got robbed of {item} by {suspect}",
        "person blocked my way pulled {weapon} and demanded my {item}", "{suspect} wrestled {item} away from {victim} using violence",
    ],
    "Burglary": [
        "somebody broke house lock entered and took {item}", "we not at home thief came inside through window stole {item}",
        "{suspect} broke into {victim}'s house and stole {item}", "shop shutter broken everything inside gone",
        "my room door lock broken {item} stolen from inside", "{suspect} entered locked flat of {victim} through broken door",
        "someone opened window climbed into our home and took {item}", "locked office broken into and {item} missing",
        "{victim}'s shop was broken into by {suspect}", "thieves entered house when family away and removed {item}",
        "{suspect} forced entry into {victim}'s home to steal {item}", "someone breaking the back door entered our closed shop for stealing",
        "came back house ransacked door forced open {item} missing", "{suspect} climbed into {victim}'s apartment through a window to steal",
        "our locked store entered by breaking shutter and stock stolen", "{victim} reported {suspect} broke into the house last night",
        "unoccupied home window smashed someone went inside taking {item}", "{suspect} pried open {victim}'s locked premises and emptied a cupboard",
    ],
    "Assault": [
        "{suspect} hit {victim} with {weapon}", "{suspect} beat {victim} badly near road",
        "one person punched me on face now bleeding", "{victim} was beaten by {suspect}",
        "{suspect} slapped and kicked {victim}", "he hitting me and i got injury please help",
        "{victim} got attacked by {suspect} using {weapon}", "fight happened they beat me with {weapon} nothing taken",
        "{suspect} attacked {victim} after argument", "{suspect} stabbed {victim} with {weapon}",
        "i am {victim} and {suspect} hit me", "{suspect} is beating {victim} and causing injuries",
        "{suspect} punched {victim} several times after quarrel", "{victim} got hit by {suspect} and needed doctor",
        "{suspect} kicked {victim} to the ground during fight", "{victim} says {suspect} attacked them without taking anything",
        "{suspect} struck {victim} repeatedly causing bruises", "{victim} suffered injuries when {suspect} hit them with {weapon}",
    ],
    "Fraud": [
        "paid money for {item} seller disappeared never delivered", "{suspect} cheated {victim} with fake investment and took money",
        "someone calling bank manager asked transfer money i paid now no response", "{victim} sent advance payment to {suspect} who gave fake receipt",
        "online seller took advance but goods not sent he blocked my number", "{suspect} sold fake gold to {victim}",
        "job offer fake they collected fee then disappeared", "{victim} was cheated by {suspect} promising double money",
        "fake rental agreement he took deposit and vanished", "{suspect} tricked {victim} into paying for nonexistent service",
        "they made forged signature and withdrew my money", "{suspect} promised {victim} guaranteed return but stole the investment",
        "sent payment after false promise of delivery and seller vanished", "{suspect} conned {victim} with forged documents for money",
        "caller claiming from bank fooled me into sending funds", "{victim} lost savings after {suspect} promised a fake job",
        "gave a deposit for goods that never existed seller refuses refund", "{suspect} collected money from {victim} using fabricated bills",
    ],
    "Cybercrime": [
        "my account hacked password changed without me", "{suspect} hacked {victim}'s email and copied private files",
        "i clicked fake link now someone accessing my bank account", "{victim}'s computer infected with virus by {suspect}",
        "all files locked ransom message asking bitcoin", "{suspect} sent phishing link to {victim} and stole password",
        "someone hacked whatsapp and sending messages as me", "{victim} says {suspect} installed malware on the laptop",
        "otp shared on fake website now account taken over", "{suspect} broke into {victim}'s online account without permission",
        "my computer data encrypted after opening email attachment", "{suspect} stole {victim}'s login credentials using phishing",
        "unknown login on my email and password reset by someone", "{suspect} used a malicious attachment to compromise {victim}'s device",
        "office server hacked private records downloaded", "{victim} received ransomware sent by {suspect}",
        "a counterfeit login website captured my password and accessed my account", "{suspect} encrypted {victim}'s files and demanded digital ransom",
    ],
    "Vandalism": [
        "{suspect} smashed {victim}'s car window and left nothing stolen", "someone scratched my vehicle on purpose",
        "{suspect} broke {victim}'s shop sign for no reason", "people sprayed graffiti on public wall",
        "{victim}'s bike damaged by {suspect} intentionally", "someone broke bus shelter glass nothing taken",
        "{suspect} deliberately damaged {victim}'s {item}", "public benches broken and street lights smashed",
        "{suspect} threw stones at {victim}'s parked car and broke mirror", "someone painted dirty words on our gate",
        "{victim}'s wall defaced by {suspect}", "they punctured tyres and scratched car intentionally",
        "{suspect} damaged {victim}'s windows without entering or stealing", "someone tore public sign and smashed nearby lights",
        "{victim} says {suspect} deliberately broke the shop board", "my parked vehicle damaged with paint and scratches",
        "{suspect} defaced {victim}'s fence using spray paint", "empty bus stop glass shattered on purpose nothing removed",
    ],
    "Arson": [
        "{suspect} poured petrol and set {victim}'s shop on fire", "somebody intentionally burnt my vehicle",
        "{suspect} set fire to {victim}'s house on purpose", "they put fuel on shed and lit it deliberately",
        "{victim}'s car was set on fire by {suspect}", "someone burning garbage against house to start fire",
        "{suspect} deliberately ignited {victim}'s storage room", "man poured kerosene on door then lighted match",
        "{suspect} torched {victim}'s building", "our shop burnt by someone using petrol",
        "{victim} saw {suspect} intentionally starting a fire", "someone put burning cloth inside empty warehouse on purpose",
        "{suspect} lit fuel beside {victim}'s home to burn it", "saw person deliberately setting parked car alight",
        "{victim}'s store was deliberately burned down by {suspect}", "they splashed petrol over shutter and started blaze",
        "{suspect} used an accelerant to ignite {victim}'s garage", "a person deliberately started flames inside the empty kiosk",
    ],
}

# Training-only nuisance text prevents memorizing crime words mentioned solely
# in negations or unrelated context. It never changes the labeled incident.
DISTRACTORS = ["no weapon was used", "nobody was injured", "i cannot identify the person",
               "police asked for details", "please help me sir", "kindly take my complaint",
               "this happened suddenly", "i am very worried", "there was no fire",
               "nobody hacked any account", "no one broke into a house", "no payment was made"]
TYPOS = {"stolen": "stoln", "mobile": "moblie", "phone": "fone", "money": "mony", "account": "acount",
         "hacked": "hackd", "someone": "somone", "somebody": "sombody", "threatened": "thretend",
         "snatched": "snachd", "knife": "nife", "house": "hous", "broke": "brok", "broken": "brokn",
         "cheated": "cheted", "received": "recieved", "password": "pasword", "window": "windw",
         "injured": "injurd", "payment": "paymnt", "deliberately": "delibrately", "yesterday": "ystrday",
         "with": "wid", "from": "frm", "please": "pls", "before": "b4", "people": "ppl", "night": "nite"}


def perturb(text, entities, rng, style):
    """Apply casing/spacing/typos and transport every gold span exactly.

    There is no inference-time autocorrect: unknown names must stay intact.
    Names/places retain their spelling; property/weapon/evidence mentions can
    have typos, whose offsets are transported with the rest of the text.
    """
    pieces, mapping, position = [], {}, 0
    for match in re.finditer(r"\w+|[^\w]", text, re.UNICODE):
        value = match.group()
        containing = next((e for e in entities if e["start"] <= match.start() < e["end"]), None)
        protected = containing is not None and containing["label"] in {"SUSPECT", "VICTIM", "LOCATION", "DATE", "TIME"}
        if style in {1, 2, 3}:
            value = value.lower()
        elif style == 4:
            value = value.upper()
        if style in {2, 3, 5} and not protected and value.isalpha():
            low = value.lower()
            if low in TYPOS and rng.random() < .6:
                value = TYPOS[low]
            elif len(value) >= 5 and rng.random() < .09:
                j = rng.randrange(1, len(value) - 1)
                value = value[:j] + value[j + 1:]
        if style == 3 and value in {",", ".", ";"} and containing is None:
            value = " "
        if style in {2, 3} and containing is None and value in {"a", "the", "was", "is"} and rng.random() < .12:
            value = ""
        if style == 5 and value == " ":
            value = rng.choice([" ", "  ", "\n"])
        mapping[match.start()] = position
        pieces.append(value)
        position += len(value)
        mapping[match.end()] = position
    result = "".join(pieces)
    spans = [{"start": mapping[e["start"]], "end": mapping[e["end"]], "label": e["label"]} for e in entities]
    for ent in spans:
        ent["text"] = result[ent["start"]:ent["end"]]
    return result, spans


def make_informal_report(index, rng=None):
    from .generate import render
    # Each row has an independent seed, making dataset regeneration and probes
    # independent of the order in which callers request rows.
    rng = rng or random.Random(SEED + 7919 * index)
    category = CATEGORIES[index % 8]
    variant = (index // 8) % 18
    split = "train" if variant < 12 else "validation" if variant < 14 else "development" if variant < 16 else "test"
    suspect, victim = rng.sample(NAMES[split], 2)
    # Invented out-of-vocabulary names teach context rather than a whitelist.
    if rng.random() < .40:
        def invented():
            return " ".join("".join(rng.choice(["qa", "zu", "ven", "lo", "mi", "tse", "no", "ri", "ska", "ba", "ha", "ya", "sa", "ti", "ko", "de", "la", "ne"]) for _ in range(rng.randint(2, 4))).title()
                            for _ in range(rng.choice([1, 1, 1, 2, 2, 3])))
        suspect, victim = invented(), invented()
    values = {"suspect": suspect, "victim": victim, "location": rng.choice(PLACES[split]),
              "item": rng.choice(ITEMS), "weapon": rng.choice(WEAPONS)}
    template = EVENTS[category][variant]
    if rng.random() < .5:
        template = template.replace("with {weapon}", "with a {weapon}").replace("using {weapon}", "using a {weapon}")
    # Sometimes attach no explicit names/places: the model must accept omission.
    if rng.random() < .65:
        template += rng.choice([" near {location}", " at {location}", " in {location}"])
    if rng.random() < .2:
        if "{victim}" not in template:
            template += rng.choice([". my name is {victim}", ". complainant {victim}", ". victim name {victim}"])
        if "{suspect}" not in template and rng.random() < .6:
            template += rng.choice([". suspect name {suspect}", ". i suspect {suspect}", ". accused {suspect}"])
    if rng.random() < .15:
        template = "sir " + template
    if rng.random() < .18:
        # Only attach negation distractors that cannot contradict this event.
        choices = DISTRACTORS[2:8]
        template += ". " + rng.choice(choices)
    date = datetime(2023, 1, 1) + timedelta(days=rng.randrange(1096), minutes=rng.randrange(1440))
    if rng.random() < .32:
        values.update(date=rng.choice([f"{date:%Y-%m-%d}", f"{date:%d/%m/%Y}", "yesterday", "last night", "today morning"]),
                      time=rng.choice([f"{date:%H:%M}", "8 pm", "10 am", "midnight"]))
        template += " on {date} at {time}"
    if rng.random() < .15:
        values["evidence"] = rng.choice(["CCTV footage", "screenshots", "camera recording", "transaction receipt", "witness statement"])
        template += ". i have {evidence}"
    text, entities = render(template, values)
    style = (index // 144) % 6
    text, entities = perturb(text, entities, rng, style)
    return {"report_id": f"INF-{index:07d}", "narrative": text, "crime_type": category,
            "reported_at": date.isoformat(timespec="seconds"), "district": values["location"],
            "template_group": f"informal-{category}-{variant}", "split": split,
            "source": "synthetic-informal-v3", "entities_json": json.dumps(entities, ensure_ascii=False)}


def informal_frame(count=14400):
    import pandas as pd
    return pd.DataFrame([make_informal_report(i) for i in range(count)])
