# Comprehensive language mapping for African languages
# Includes ISO 639-1 (2-letter), ISO 639-2/3 (3-letter), and NLLB language codes

LANGUAGE_MAPPING = {
    # West African languages
    "twi": {
        "iso2": "tw",
        "iso3": "twi",
        "name": "Twi",
        "nllb_code": "twi_Latn",
        "script": "Latn"
    },
    "yor": {
        "iso2": "yo",
        "iso3": "yor",
        "name": "Yoruba",
        "nllb_code": "yor_Latn",
        "script": "Latn"
    },
    "hau": {
        "iso2": "ha",
        "iso3": "hau",
        "name": "Hausa",
        "nllb_code": "hau_Latn",
        "script": "Latn"
    },
    "ibo": {
        "iso2": "ig",
        "iso3": "ibo",
        "name": "Igbo",
        "nllb_code": "ibo_Latn",
        "script": "Latn"
    },
    "ewe": {
        "iso2": "ee",
        "iso3": "ewe",
        "name": "Ewe",
        "nllb_code": "ewe_Latn",
        "script": "Latn"
    },
    "fon": {
        "iso2": "fon",
        "iso3": "fon",
        "name": "Fon",
        "nllb_code": "fon_Latn",
        "script": "Latn"
    },
    
    # East African languages
    "swa": {
        "iso2": "sw",
        "iso3": "swa",
        "name": "Swahili",
        "nllb_code": "swh_Latn",  # Note: NLLB uses swh for Swahili
        "script": "Latn"
    },
    "amh": {
        "iso2": "am",
        "iso3": "amh",
        "name": "Amharic",
        "nllb_code": "amh_Ethi",
        "script": "Ethi"
    },
    "orm": {
        "iso2": "om",
        "iso3": "orm",
        "name": "Oromo",
        "nllb_code": "gaz_Latn",  # Note: NLLB uses gaz for West Central Oromo
        "script": "Latn"
    },
    "som": {
        "iso2": "so",
        "iso3": "som",
        "name": "Somali",
        "nllb_code": "som_Latn",
        "script": "Latn"
    },
    "tir": {
        "iso2": "ti",
        "iso3": "tir",
        "name": "Tigrinya",
        "nllb_code": "tir_Ethi",
        "script": "Ethi"
    },
    
    # Southern African languages
    "zul": {
        "iso2": "zu",
        "iso3": "zul",
        "name": "Zulu",
        "nllb_code": "zul_Latn",
        "script": "Latn"
    },
    "xho": {
        "iso2": "xh",
        "iso3": "xho",
        "name": "Xhosa",
        "nllb_code": "xho_Latn",
        "script": "Latn"
    },
    "sot": {
        "iso2": "st",
        "iso3": "sot",
        "name": "Southern Sotho",
        "nllb_code": "sot_Latn",
        "script": "Latn"
    },
    "tsn": {
        "iso2": "tn",
        "iso3": "tsn",
        "name": "Tswana",
        "nllb_code": "tsn_Latn",
        "script": "Latn"
    },
    "ven": {
        "iso2": "ve",
        "iso3": "ven",
        "name": "Venda",
        "nllb_code": "ven_Latn",
        "script": "Latn"
    },
    "nso": {
        "iso2": "nso",
        "iso3": "nso",
        "name": "Northern Sotho",
        "nllb_code": "nso_Latn",
        "script": "Latn"
    },
    "tso": {
        "iso2": "ts",
        "iso3": "tso",
        "name": "Tsonga",
        "nllb_code": "tso_Latn",
        "script": "Latn"
    },
    
    # North African languages
    "ara": {
        "iso2": "ar",
        "iso3": "ara",
        "name": "Arabic",
        "nllb_code": "arb_Arab",  # Modern Standard Arabic
        "script": "Arab"
    },
    "ber": {
        "iso2": "ber",
        "iso3": "ber",
        "name": "Berber",
        "nllb_code": "ber_Latn",  # Generic Berber code
        "script": "Latn"
    },
    
    # European languages commonly used in Africa
    "eng": {
        "iso2": "en",
        "iso3": "eng",
        "name": "English",
        "nllb_code": "eng_Latn",
        "script": "Latn"
    },
    "fra": {
        "iso2": "fr",
        "iso3": "fra",
        "name": "French",
        "nllb_code": "fra_Latn",
        "script": "Latn"
    },
    "por": {
        "iso2": "pt",
        "iso3": "por",
        "name": "Portuguese",
        "nllb_code": "por_Latn",
        "script": "Latn"
    },
    "spa": {
        "iso2": "es",
        "iso3": "spa",
        "name": "Spanish",
        "nllb_code": "spa_Latn",
        "script": "Latn"
    }
}

def get_language_info(lang_code):
    """Get complete language information from any code (iso2, iso3, or name)"""
    # First try exact match
    for code, info in LANGUAGE_MAPPING.items():
        if lang_code.lower() in [code, info["iso2"], info["iso3"], info["name"].lower()]:
            return info
    
    # If not found, try to match the beginning of the name
    for code, info in LANGUAGE_MAPPING.items():
        if info["name"].lower().startswith(lang_code.lower()):
            return info
    
    # If still not found, return None
    return None

def get_nllb_code(lang_code):
    """Get NLLB language code from any code (iso2, iso3, or name)"""
    info = get_language_info(lang_code)
    if info:
        return info["nllb_code"]
    else:
        # Default to Latin script if not found
        return f"{lang_code}_Latn"

def get_iso3_code(lang_code):
    """Get ISO 639-3 code from any code (iso2, iso3, or name)"""
    info = get_language_info(lang_code)
    if info:
        return info["iso3"]
    else:
        return lang_code  # Return as-is if not found

def get_iso2_code(lang_code):
    """Get ISO 639-1 code from any code (iso2, iso3, or name)"""
    info = get_language_info(lang_code)
    if info:
        return info["iso2"]
    else:
        return lang_code  # Return as-is if not found

def get_language_name(lang_code):
    """Get language name from any code (iso2, iso3, or name)"""
    info = get_language_info(lang_code)
    if info:
        return info["name"]
    else:
        return lang_code  # Return as-is if not found