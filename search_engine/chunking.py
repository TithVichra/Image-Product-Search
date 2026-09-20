"""
Product Description Chunking & Semantic Context Module
Enriches product metadata with:
1. N-Gram extraction (unigrams, bigrams, trigrams) of key fashion attributes.
2. Semantic lexicon and synonym context expansion (e.g., jeans <-> denim trousers, sneakers <-> athletic shoes).
3. Natural descriptive photo-captions tailored to CLIP vector attention.
"""

import re
from typing import List, Dict, Any, Set


# Fashion Domain Knowledge Base & Contextual Synonyms
FASHION_LEXICON: Dict[str, Dict[str, Any]] = {
    "jeans": {
        "synonyms": ["jean", "jeans", "denim", "denim pants", "skinny jeans", "ripped jeans", "distressed jeans"],
        "context": "casual denim bottomwear"
    },
    "trousers": {
        "synonyms": ["trouser", "trousers", "pant", "pants", "chinos", "chino", "slacks", "formal trousers", "cargo pants", "khakis", "bottomwear"],
        "context": "tailored bottomwear trousers pants"
    },
    "shorts": {
        "synonyms": ["short", "shorts", "bermuda", "bermudas", "cargo shorts", "denim shorts", "sweat shorts", "bottomwear"],
        "context": "casual summer bottomwear shorts"
    },
    "track pants": {
        "synonyms": ["track pants", "track pant", "trackpants", "sweatpants", "sweat pants", "joggers", "jogger", "track trousers"],
        "context": "athletic lounge bottomwear track pants"
    },
    "tshirts": {
        "synonyms": ["t-shirt", "t-shirts", "tshirt", "tshirts", "tee", "tees", "crewneck", "v-neck", "polo shirt", "polo", "graphic tee"],
        "context": "casual topwear t-shirt"
    },
    "shirts": {
        "synonyms": ["shirt", "shirts", "dress shirt", "button-down", "formal shirt", "casual shirt", "oxford shirt", "topwear"],
        "context": "collared button-down topwear shirt"
    },
    "tops": {
        "synonyms": ["top", "tops", "crop top", "blouse", "blouses", "camisole top", "peplum top", "topwear"],
        "context": "women's topwear blouse top"
    },
    "sweatshirts": {
        "synonyms": ["sweatshirt", "sweatshirts", "hoodie", "hoodies", "pullover", "fleece", "hooded sweatshirt"],
        "context": "casual warm topwear sweatshirt hoodie"
    },
    "sweaters": {
        "synonyms": ["sweater", "sweaters", "cardigan", "cardigans", "knitwear", "jumper", "knitted pullover"],
        "context": "warm knitted winter sweater cardigan"
    },
    "jackets": {
        "synonyms": ["jacket", "jackets", "outerwear", "coat", "coats", "blazer", "blazers", "windbreaker", "bomber jacket", "biker jacket", "leather jacket"],
        "context": "outerwear layer jacket coat"
    },
    "dresses": {
        "synonyms": ["dress", "dresses", "dresse", "gown", "gowns", "frock", "frocks", "one-piece", "sundress", "cocktail dress", "maxi dress", "midi dress"],
        "context": "women's fashion dress gown"
    },
    "kurtas": {
        "synonyms": ["kurta", "kurtas", "kurti", "kurtis", "tunic", "tunics", "ethnic tunic", "anarkali"],
        "context": "traditional ethnic tunic kurta"
    },
    "sarees": {
        "synonyms": ["saree", "sarees", "sari", "saris", "traditional sari"],
        "context": "traditional ethnic draped saree"
    },
    "skirts": {
        "synonyms": ["skirt", "skirts", "mini skirt", "maxi skirt", "pleated skirt"],
        "context": "women's bottomwear skirt"
    },
    "jumpsuits": {
        "synonyms": ["jumpsuit", "jumpsuits", "romper", "rompers", "dungarees", "playsuit", "one-piece suit"],
        "context": "full body one-piece jumpsuit"
    },
    "leggings": {
        "synonyms": ["legging", "leggings", "tights", "churidar", "churidars", "jeggings", "capri", "capris"],
        "context": "elastic stretch bottomwear leggings"
    },
    "shoes": {
        "synonyms": ["shoe", "shoes", "footwear", "sneakers", "trainers", "athletic shoes", "kicks"],
        "context": "footwear shoes sneakers"
    },
    "sports shoes": {
        "synonyms": ["sports shoes", "running shoes", "running shoe", "trainers", "athletic footwear", "sneakers", "jogging shoes", "gym shoes"],
        "context": "athletic sports running footwear"
    },
    "casual shoes": {
        "synonyms": ["casual shoes", "loafers", "loafer", "slip-on", "slip-ons", "boat shoes", "canvas shoes"],
        "context": "everyday casual footwear shoes"
    },
    "formal shoes": {
        "synonyms": ["formal shoes", "oxfords", "derbys", "derby", "brogues", "leather dress shoes", "formal footwear", "monk strap"],
        "context": "formal business dress footwear"
    },
    "sandals": {
        "synonyms": ["sandal", "sandals", "strappy sandals", "gladiator sandals", "flat sandals", "heeled sandals"],
        "context": "open strap summer footwear sandals"
    },
    "flip flops": {
        "synonyms": ["flip flop", "flip flops", "flipflop", "flipflops", "slippers", "slipper", "slides", "thongs", "chappals"],
        "context": "casual open toe flip flops slippers"
    },
    "heels": {
        "synonyms": ["heels", "heel", "high heels", "stilettos", "pumps", "wedges", "wedge", "heeled footwear"],
        "context": "elevated heeled women's footwear"
    },
    "flats": {
        "synonyms": ["flats", "flat", "ballerinas", "ballerina", "loafers", "slip-ons", "flat shoes", "bellies"],
        "context": "flat sole women's footwear"
    },
    "socks": {
        "synonyms": ["sock", "socks", "ankle socks", "crew socks", "cotton socks", "athletic socks", "hosiery"],
        "context": "hosiery foot wear socks"
    },
    "watches": {
        "synonyms": ["watch", "watches", "watche", "timepiece", "wrist watch", "chronograph", "analog watch", "digital watch"],
        "context": "wrist timepiece accessory watch"
    },
    "sunglasses": {
        "synonyms": ["sunglasses", "sunglass", "sunglasse", "shades", "eyewear", "sun glasses", "goggles", "aviators", "wayfarers"],
        "context": "eye protection fashion eyewear sunglasses"
    },
    "handbags": {
        "synonyms": ["handbag", "handbags", "purse", "shoulder bag", "tote", "tote bag", "satchel", "crossbody bag", "crossbody"],
        "context": "fashion accessory handbag purse"
    },
    "backpacks": {
        "synonyms": ["backpack", "backpacks", "rucksack", "bookbag", "travel pack", "knapsack", "laptop backpack"],
        "context": "travel and utility shoulder backpack"
    },
    "clutches": {
        "synonyms": ["clutch", "clutches", "clutche", "clutch bag", "pouch", "evening bag", "wristlet"],
        "context": "compact handheld evening clutch bag"
    },
    "wallets": {
        "synonyms": ["wallet", "wallets", "billfold", "coin purse", "pocketbook", "card holder", "leather wallet"],
        "context": "compact pocket personal accessory wallet"
    },
    "belts": {
        "synonyms": ["belt", "belts", "waistband", "leather belt", "buckle belt", "strap"],
        "context": "waist accessory buckle belt"
    },
    "caps": {
        "synonyms": ["cap", "caps", "hat", "hats", "baseball cap", "beanie", "sun hat", "headwear", "snapback"],
        "context": "headwear accessory cap hat"
    },
    "ties": {
        "synonyms": ["tie", "ties", "necktie", "bow tie", "formal tie", "cravat"],
        "context": "formal neckwear tie"
    },
    "earrings": {
        "synonyms": ["earring", "earrings", "studs", "hoops", "drop earrings", "danglers", "jhumkas"],
        "context": "ear jewellery earrings"
    },
    "necklaces": {
        "synonyms": ["necklace", "necklaces", "pendant", "pendants", "chain", "chains", "locket", "choker"],
        "context": "neck jewellery necklace pendant"
    },
    "bracelets": {
        "synonyms": ["bracelet", "bracelets", "bangle", "bangles", "wristlet", "wristband", "kada"],
        "context": "wrist jewellery bracelet bangle"
    },
    "rings": {
        "synonyms": ["ring", "rings", "finger ring", "band", "diamond ring"],
        "context": "finger jewellery ring"
    },
    "cufflinks": {
        "synonyms": ["cufflinks", "cufflink", "cuff links"],
        "context": "formal shirt accessory cufflinks"
    },
    "scarves": {
        "synonyms": ["scarf", "scarves", "scarve", "stole", "stoles", "muffler", "wrap", "shawl"],
        "context": "neckwear fashion accessory scarf stole"
    },
    "dupattas": {
        "synonyms": ["dupatta", "dupattas", "chunni", "chunari", "traditional stole"],
        "context": "ethnic traditional draped dupatta"
    },
    "bra": {
        "synonyms": ["bra", "bras", "brassiere", "sports bra", "padded bra", "lingerie bra"],
        "context": "intimate innerwear support bra"
    },
    "briefs": {
        "synonyms": ["brief", "briefs", "underwear", "boxers", "boxer briefs", "trunks", "underpants", "panties", "panty"],
        "context": "intimate innerwear briefs underwear"
    },
    "nightwear": {
        "synonyms": ["nightdress", "night suit", "night suits", "nightwear", "sleepwear", "pajamas", "pyjamas", "lounge wear", "loungewear"],
        "context": "comfortable sleeping lounge nightwear"
    },
    "swimwear": {
        "synonyms": ["swimwear", "swimsuit", "bathing suit", "swimming trunks", "bikini", "swim shorts"],
        "context": "swimming athletic beach swimwear"
    },
    "perfumes": {
        "synonyms": ["perfume", "perfumes", "fragrance", "scent", "eau de parfum", "cologne", "body mist", "mist"],
        "context": "scent fragrance perfume"
    },
    "deodorants": {
        "synonyms": ["deodorant", "deodorants", "body spray", "anti-perspirant", "deodorant spray", "fragrance spray"],
        "context": "personal care deodorant spray"
    },
    "lipstick": {
        "synonyms": ["lipstick", "lipsticks", "lip color", "lip colour", "lip shade", "lip gloss", "lipgloss", "lip lacquer"],
        "context": "cosmetic lip makeup lipstick"
    },
    "nail polish": {
        "synonyms": ["nail polish", "nail lacquer", "nail enamel", "nail paint"],
        "context": "cosmetic nail care polish lacquer"
    }
}

# Strictly incompatible categories to prevent cross-contamination (e.g. dress vs shoes)
INCOMPATIBLE_CATEGORIES: Dict[str, Set[str]] = {
    "dresses": {"tshirts", "shirts", "shoes", "sports shoes", "casual shoes", "formal shoes", "watches", "handbags", "backpacks", "wallets", "belts", "perfumes", "deodorants", "jeans"},
    "tshirts": {"dresses", "formal shoes", "watches", "handbags", "backpacks", "wallets", "belts", "perfumes", "deodorants"},
    "shirts": {"dresses", "sports shoes", "handbags", "backpacks", "wallets", "belts", "perfumes", "deodorants"},
    "jeans": {"dresses", "watches", "handbags", "backpacks", "wallets", "belts", "perfumes", "deodorants"},
    "shoes": {"tshirts", "shirts", "dresses", "watches", "handbags", "backpacks", "wallets", "belts", "perfumes", "deodorants"},
    "sports shoes": {"formal shoes", "dresses", "shirts", "handbags", "watches", "wallets", "belts"},
    "formal shoes": {"sports shoes", "tshirts", "dresses", "handbags", "backpacks", "wallets", "belts"},
    "watches": {"tshirts", "shirts", "dresses", "jeans", "shoes", "handbags", "backpacks", "wallets", "belts", "perfumes", "deodorants"},
    "handbags": {"tshirts", "shirts", "dresses", "jeans", "shoes", "watches", "wallets", "belts", "perfumes", "deodorants"}
}


def normalize_stem(word: str) -> str:
    """
    Normalizes common English plural and inflectional suffixes for fashion terms:
    - 'panties' -> 'panty', 'accessories' -> 'accessory'
    - 'watches' -> 'watch', 'dresses' -> 'dress', 'glasses' -> 'glass'
    - 'scarves' -> 'scarf'
    - 'socks' -> 'sock', 'bras' -> 'bra', 'lipsticks' -> 'lipstick', 'flats' -> 'flat'
    """
    w = word.lower().strip()
    if len(w) <= 3:
        return w
    if w == "scarves":
        return "scarf"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ches") or w.endswith("shes") or w.endswith("xes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and not w.endswith("us") and not w.endswith("is"):
        return w[:-1]
    return w


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^\w\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_word_tokens(text: str) -> List[str]:
    """
    Extracts lowercased word tokens from text.
    """
    cleaned = clean_text(text).lower()
    return [w for w in cleaned.split() if len(w) > 1]


def extract_normalized_token_set(text: str) -> Set[str]:
    """
    Returns a set containing both raw tokens and their stemmed variants.
    """
    raw_tokens = extract_word_tokens(text)
    token_set = set(raw_tokens)
    for tok in raw_tokens:
        stem = normalize_stem(tok)
        token_set.add(stem)
    return token_set


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    Extracts contiguous n-grams from a list of tokens.
    """
    if len(tokens) < n or n < 1:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def find_matching_lexicon_entry(category_str: str) -> Dict[str, Any]:
    """
    Finds the best matching semantic lexicon entry for a product category or article type.
    """
    cat_lower = category_str.lower().strip()
    if not cat_lower:
        return {}

    stemmed = normalize_stem(cat_lower)

    # Exact key match or stemmed match
    if cat_lower in FASHION_LEXICON:
        return FASHION_LEXICON[cat_lower]
    if stemmed in FASHION_LEXICON:
        return FASHION_LEXICON[stemmed]

    # Exact synonym match
    for key, val in FASHION_LEXICON.items():
        if cat_lower == key or stemmed == normalize_stem(key):
            return val
        for syn in val["synonyms"]:
            if cat_lower == syn or stemmed == normalize_stem(syn):
                return val

    # Substring / partial match
    for key, val in FASHION_LEXICON.items():
        if key in cat_lower or stemmed in key:
            return val
        for syn in val["synonyms"]:
            if syn in cat_lower or cat_lower in syn:
                return val

    return {}


COLOR_FAMILIES: Dict[str, Set[str]] = {
    "red": {"red", "crimson", "maroon", "ruby", "burgundy", "coral", "rust"},
    "blue": {"blue", "navy", "navy blue", "cyan", "teal", "indigo", "turquoise", "cobalt"},
    "black": {"black", "charcoal"},
    "white": {"white", "off white", "ivory"},
    "green": {"green", "olive", "emerald", "lime", "khaki", "mint", "sea green"},
    "pink": {"pink", "rose", "magenta", "peach", "fuchsia", "salmon"},
    "yellow": {"yellow", "mustard", "gold", "golden", "ochre"},
    "orange": {"orange", "tangerine", "apricot", "peach"},
    "purple": {"purple", "violet", "lavender", "mauve", "plum"},
    "brown": {"brown", "tan", "chocolate", "coffee", "bronze", "copper"},
    "grey": {"grey", "gray", "silver", "steel", "gunmetal", "charcoal", "grey melange"},
    "gray": {"grey", "gray", "silver", "steel", "gunmetal", "charcoal", "grey melange"},
    "beige": {"beige", "cream", "nude", "khaki", "tan", "sand"},
    "cream": {"cream", "beige", "off white", "ivory"}
}

NON_FASHION_TERMS: Set[str] = {
    "ball", "soccer ball", "football", "basketball", "tennis ball", "baseball", "volleyball",
    "golf ball", "bowling ball", "frisbee", "skateboard", "surfboard", "tennis racket", "baseball bat",
    "dog", "cat", "puppy", "kitten", "bird", "fish", "horse", "animal", "pet",
    "car", "truck", "motorcycle", "bicycle", "bike", "airplane", "boat", "vehicle",
    "food", "pizza", "burger", "apple", "banana", "sandwich",
    "laptop", "computer", "phone", "television", "tv", "camera"
}

FASHION_AFFINITY_KEYWORDS: Set[str] = {
    "shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandal", "sandals", "heel", "heels", "flat", "flats",
    "shirt", "shirts", "t-shirt", "tshirt", "tshirts", "tee", "tees", "top", "tops", "blouse",
    "dress", "dresses", "gown", "gowns", "frock", "skirt", "skirts", "kurta", "kurtas", "sari", "saree",
    "pant", "pants", "jean", "jeans", "trousers", "shorts", "jacket", "jackets", "coat", "blazer",
    "watch", "watches", "bag", "bags", "handbag", "handbags", "backpack", "wallet", "belt",
    "sock", "socks", "jersey", "uniform", "wear", "apparel", "clothing", "outfit", "suit", "print", "pattern"
}


def extract_search_intent(query: str) -> Dict[str, Any]:
    """
    Parses user search query to extract search intent:
    - target_gender: 'Women', 'Men', 'Unisex', or None
    - target_category: canonical key from FASHION_LEXICON or None
    - target_color: string or None
    - color_shades: Set of compatible color strings
    - is_non_fashion_query: True if query targets non-fashion entity without fashion context
    """
    clean_q = clean_text(query).lower()
    tokens = extract_word_tokens(clean_q)

    # 1. Non-fashion query detection (e.g. 'ball', 'dog', 'car')
    is_non_fashion_query = False
    non_fashion_term = None
    has_fashion_keyword = any(kw in clean_q for kw in FASHION_AFFINITY_KEYWORDS)

    for nft in sorted(NON_FASHION_TERMS, key=lambda x: len(x), reverse=True):
        if re.search(r"\b" + re.escape(nft) + r"\b", clean_q):
            if not has_fashion_keyword:
                is_non_fashion_query = True
                non_fashion_term = nft
                break

    # 2. Gender intent
    gender = None
    if re.search(r"\b(women|woman|women's|woman's|ladies|lady|female|girls|girl|for her)\b", clean_q):
        gender = "Women"
    elif re.search(r"\b(men|man|men's|man's|gentlemen|male|boys|boy|for him)\b", clean_q):
        gender = "Men"
    elif re.search(r"\bunisex\b", clean_q):
        gender = "Unisex"

    # 3. Category intent
    target_category = None
    sorted_lex_keys = sorted(FASHION_LEXICON.keys(), key=lambda x: len(x), reverse=True)
    for cat_key in sorted_lex_keys:
        cat_val = FASHION_LEXICON[cat_key]
        stemmed_cat = normalize_stem(cat_key)
        if re.search(r"\b" + re.escape(cat_key) + r"\b", clean_q) or re.search(r"\b" + re.escape(stemmed_cat) + r"\b", clean_q):
            target_category = cat_key
            break
        found_syn = False
        for syn in sorted(cat_val["synonyms"], key=lambda s: len(s), reverse=True):
            stemmed_syn = normalize_stem(syn)
            if re.search(r"\b" + re.escape(syn) + r"\b", clean_q) or re.search(r"\b" + re.escape(stemmed_syn) + r"\b", clean_q):
                if syn == "dress" and "dress shirt" in clean_q:
                    continue
                target_category = cat_key
                found_syn = True
                break
        if found_syn:
            break

    # If category is inherently gendered and no explicit gender was stated:
    if not gender and target_category in {"dresses", "heels", "flats", "bra", "skirts", "sarees", "dupattas", "lipstick", "nail polish"}:
        gender = "Women"

    # 4. Color extraction & color family mapping
    common_colors = [
        "red", "blue", "navy", "black", "white", "green", "yellow", "pink", "purple",
        "orange", "brown", "grey", "gray", "silver", "gold", "maroon", "beige", "cream"
    ]
    matched_color = None
    for c in common_colors:
        if re.search(r"\b" + c + r"\b", clean_q):
            matched_color = c
            break

    color_shades = set()
    if matched_color:
        color_shades = COLOR_FAMILIES.get(matched_color, {matched_color})
        # Add the base word itself
        color_shades.add(matched_color)

    return {
        "target_category": target_category,
        "target_gender": gender,
        "target_color": matched_color,
        "color_shades": sorted(list(color_shades)),
        "is_non_fashion_query": is_non_fashion_query,
        "non_fashion_term": non_fashion_term,
        "tokens": tokens,
        "clean_query": clean_q
    }


def generate_attribute_ngrams(product_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generates structured, attribute-bound 1-grams, 2-grams, and 3-grams from product metadata.
    Ensures colors, article types, and genders are bound in meaningful context.
    """
    title = clean_text(product_data.get("productDisplayName", "")).lower()
    gender = clean_text(product_data.get("gender", "")).lower()
    color = clean_text(product_data.get("baseColour", "")).lower()
    article = clean_text(product_data.get("articleType", "")).lower()
    usage = clean_text(product_data.get("usage", "")).lower()

    unigrams: Set[str] = set()
    bigrams: Set[str] = set()
    trigrams: Set[str] = set()

    # Title n-grams
    title_tokens = extract_word_tokens(title)
    for tok in title_tokens:
        unigrams.add(tok)
    for bg in extract_ngrams(title_tokens, 2):
        bigrams.add(bg)
    for tg in extract_ngrams(title_tokens, 3):
        trigrams.add(tg)

    # Core attribute bindings
    if color and article:
        bigrams.add(f"{color} {article}")
    if gender and article and gender != "unisex":
        bigrams.add(f"{gender} {article}")
    if usage and article:
        bigrams.add(f"{usage} {article}")
    if color and usage:
        bigrams.add(f"{color} {usage}")

    if gender and color and article and gender != "unisex":
        trigrams.add(f"{gender} {color} {article}")
    if usage and color and article:
        trigrams.add(f"{usage} {color} {article}")
    if gender and usage and article and gender != "unisex":
        trigrams.add(f"{gender} {usage} {article}")

    # Add synonyms from lexicon
    lex = find_matching_lexicon_entry(article)
    if lex:
        for syn in lex.get("synonyms", []):
            unigrams.add(syn)
            if color:
                bigrams.add(f"{color} {syn}")
            if gender and gender != "unisex":
                bigrams.add(f"{gender} {syn}")
            if color and gender and gender != "unisex":
                trigrams.add(f"{gender} {color} {syn}")

    return {
        "unigrams": sorted(list(unigrams)),
        "bigrams": sorted(list(bigrams)),
        "trigrams": sorted(list(trigrams))
    }


def chunk_product_description(product_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Given a product dict containing fields from styles.csv and/or style json:
      - id, productDisplayName, gender, masterCategory, subCategory,
      - articleType, baseColour, season, year, usage, description
    Returns a list of high-value semantic and n-gram chunk objects:
      [{ "chunk_id": int, "chunk_type": str, "text": str }]
    """
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    title = clean_text(product_data.get("productDisplayName", ""))
    gender = clean_text(product_data.get("gender", ""))
    master_cat = clean_text(product_data.get("masterCategory", ""))
    sub_cat = clean_text(product_data.get("subCategory", ""))
    article_type = clean_text(product_data.get("articleType", ""))
    color = clean_text(product_data.get("baseColour", ""))
    season = clean_text(product_data.get("season", ""))
    year = clean_text(product_data.get("year", ""))
    usage = clean_text(product_data.get("usage", ""))
    description = clean_text(product_data.get("description", ""))

    # Chunk 1: Primary Natural Description (Photo caption style for CLIP alignment)
    caption_parts = []
    if color:
        caption_parts.append(color.lower())
    if gender and gender.lower() != "unisex":
        caption_parts.append(f"{gender.lower()}'s")
    if article_type:
        caption_parts.append(article_type.lower())
    elif sub_cat:
        caption_parts.append(sub_cat.lower())

    desc_phrase = " ".join(caption_parts)
    if title:
        if desc_phrase and desc_phrase not in title.lower():
            natural_chunk1 = f"A photo of {title}, a {desc_phrase}."
        else:
            natural_chunk1 = f"A photo of {title}."
    else:
        natural_chunk1 = f"A photo of a {desc_phrase}."

    chunks.append({
        "chunk_id": chunk_id,
        "chunk_type": "identity_category",
        "text": natural_chunk1
    })
    chunk_id += 1

    # Chunk 2: Style, Occasion, and Season attributes
    attr_clauses = []
    if color:
        attr_clauses.append(f"{color.lower()} colored")
    if usage:
        attr_clauses.append(f"designed for {usage.lower()} wear")
    if season or year:
        season_str = " ".join(filter(None, [season.lower(), str(year)]))
        attr_clauses.append(f"suitable for {season_str}")
    if article_type:
        attr_clauses.append(f"{article_type.lower()} fashion product")

    if attr_clauses:
        natural_chunk2 = f"{title}. " + ", ".join(attr_clauses) + "."
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_type": "style_attributes",
            "text": natural_chunk2
        })
        chunk_id += 1

    # Chunk 3: Structured N-Gram Chunk (Direct Bigram & Trigram Phrase Representation)
    ngram_data = generate_attribute_ngrams(product_data)
    key_bigrams = ngram_data["bigrams"][:6]
    key_trigrams = ngram_data["trigrams"][:4]
    ngram_phrases = key_bigrams + key_trigrams
    if ngram_phrases:
        ngram_text = "Key product phrases and n-grams: " + ", ".join(ngram_phrases) + "."
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_type": "ngram_context",
            "text": ngram_text
        })
        chunk_id += 1

    # Chunk 4: Semantic Context & Taxonomy Meaning Expansion
    lex = find_matching_lexicon_entry(article_type) or find_matching_lexicon_entry(sub_cat)
    context_parts = []
    if master_cat or sub_cat:
        cat_hierarchy = " > ".join(filter(None, [master_cat, sub_cat, article_type]))
        context_parts.append(f"Category hierarchy: {cat_hierarchy}")
    if lex:
        syn_str = ", ".join(lex["synonyms"][:4])
        context_parts.append(f"Item meaning and synonyms: {syn_str}")
        context_parts.append(f"Styling context: {lex['context']}")
    if gender:
        context_parts.append(f"Target audience: {gender}")

    if context_parts:
        semantic_text = f"Contextual fashion ontology for {title}: " + ". ".join(context_parts) + "."
        chunks.append({
            "chunk_id": chunk_id,
            "chunk_type": "semantic_context",
            "text": semantic_text
        })
        chunk_id += 1

    # Chunk 5: Detailed Description Sentences (if present in style JSON)
    if description and description.lower() != title.lower():
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", description) if s.strip()]
        for sent in sentences[:2]:
            if len(sent) > 10:
                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_type": "description_detail",
                    "text": sent
                })
                chunk_id += 1

    # Fallback if no chunks created
    if not chunks:
        chunks.append({
            "chunk_id": 0,
            "chunk_type": "generic",
            "text": title or f"Fashion item {product_data.get('id', '')}"
        })

    return chunks
