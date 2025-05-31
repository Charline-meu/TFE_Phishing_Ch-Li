import re
import string
import pandas as pd
from email import message_from_string, policy
from email.message import EmailMessage
from bs4 import BeautifulSoup, Comment
import numpy as np
from urllib.parse import urlparse
import time
from urllib.parse import urlparse, urlunparse
import base64
import quopri

def load_email_from_string(email_str: str) -> EmailMessage:
    return message_from_string(email_str, policy=policy.default)

def extract_section_statistics(msg: EmailMessage,html_content:str) -> dict:
    """
    Extract the features in the section statistics category, such as:
    Number of text/plain sections
    Number of text/html sections
    Number of image sections
    Number of application sections
    Ratio of text/plain to any text sections
    Length of text in text/html section 
    """
    num_text_plain = 0
    num_text_html = 0
    num_image_sections = 0
    num_application_sections = 0
    length_text_html = 0

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type().lower()
            if ctype.startswith("image/"):
                num_image_sections += 1
            elif ctype.startswith("application/"):
                num_application_sections += 1
            elif ctype == "text/plain":
                num_text_plain += 1
            elif ctype == "text/html":
                num_text_html += 1
                length_text_html += len(html_content)
    else:
        # Non-multipart message
        ctype = msg.get_content_type().lower()
        if ctype.startswith("image/"):
            num_image_sections = 1
        elif ctype.startswith("application/"):
            num_application_sections = 1
        elif ctype == "text/plain":
            num_text_plain = 1
        elif ctype == "text/html":
            num_text_html = 1
            length_text_html = len(html_content)
    total_text_sections = num_text_plain + num_text_html
    ratio_text_plain_to_any = (num_text_plain / total_text_sections) if total_text_sections > 0 else 0

    return {
        'number_of_text_plain_sections': num_text_plain,
        'number_of_text_html_sections': num_text_html,
        'number_of_image_sections': num_image_sections,
        'number_of_application_sections': num_application_sections,
        'ratio_text_plain_to_any_text_sections': ratio_text_plain_to_any,
        'length_of_text_in_text_html_section': length_text_html
    }


def extract_header_statistics(msg: EmailMessage) -> dict:

    standard_headers = [
        "From", "To", "Subject", "Date", "Message-ID",
        "MIME-Version", "Content-Type", "Content-Transfer-Encoding"
    ]
    count = 0
    for header in standard_headers:
        # Utilisation de get() permet de récupérer None si l'en-tête n'existe pas
        try:
            if msg.get(header) is not None:
                count += 1
        except Exception as e:
            continue

        
    
    return {
        'num_standart_header' : count,
        'num_in_reply_to': len(msg.get_all('In-Reply-To', [])),
        'num_received' : len(msg.get_all('Received', [])),
        'exist_x_mailer': 1 if msg.get('X-Mailer') else 0,
        'exist_user_agent' : 1 if msg.get('User-Agent') else 0,
    }

def extract_MIME_version(msg: EmailMessage) -> dict:

    mime_header = None

    for header in msg.keys():
        if header.lower() == "mime-version":  # On trouve n'importe quelle variation
            mime_header = header  # On récupère la version exacte du header

    # Si "MIME-Version" est trouvé mais avec une casse différente
    case_variance = 1 if mime_header and mime_header != "MIME-Version" else 0

    return {
        'mime_version' : 1 if msg.get('MIME-Version') else 0,
        'case_var_MV': case_variance
    }

#Needed def for the message_id features
def clean_string_for_check(s: str, allow_dot: bool = True) -> str:
    """
    Remove characters not in [a-zA-Z0-9] or '.' (optionally).
    This is used for checking if a string is hex or decimal once
    special characters are removed.
    """
    allowed = string.ascii_letters + string.digits
    if allow_dot:
        allowed += "."
    return "".join(ch for ch in s if ch in allowed)

def is_hex(s: str) -> bool:
    """
    Return True if s (after ignoring certain special chars) is purely
    hexadecimal [0-9A-Fa-f]. We convert to uppercase for simplicity.
    """
    s = s.upper()
    return bool(s) and all(ch in "0123456789ABCDEF" for ch in s)

def is_decimal(s: str) -> bool:
    """
    Return True if s is purely digits [0-9].
    """
    return bool(s) and all(ch.isdigit() for ch in s)

#End of needed def for message_id_features

def extract_message_id_features(msg: EmailMessage) -> dict:
    """
    Extract features from the Message-ID header, including:
      - ascii_boundary_char (ASCII code of '@' if present, else 0)
      - existence_of_domain (1 if there's a domain part after '@', else 0)
      - is_id_part_hex
      - is_id_part_decimal
      - existence_of_dots_in_id_part
      - existence_of_special_chars_in_id_part
      - is_domain_part_hex
      - is_domain_part_decimal
      - existence_of_dots_in_domain_part
      - existence_of_special_chars_in_domain_part
    """
    try:
        raw_msg_id = msg.get("Message-ID", "")
    #si l'ID est vide ça fait une erreur
    except Exception as e:
        return {
        "length_message_id" : 0,
        "ascii_boundary_char": 0,
        "existence_of_domain": 0,
        "is_id_part_hex": 0,
        "is_id_part_decimal": 0,
        "existence_of_dots_in_id_part": 0,
        "existence_of_special_chars_in_id_part": 0,
        "is_domain_part_hex": 0,
        "is_domain_part_decimal": 0,
        "existence_of_dots_in_domain_part": 0,
        "existence_of_special_chars_in_domain_part": 0
    } 
        
    # Remove angle brackets < > if present
    msg_id = raw_msg_id.strip("<> \t\r\n")

    

    # ASCII code for '@' if present, else 0
    ascii_boundary_char = 0

    # ID part and domain part
    id_part = ""
    domain_part = ""
    existence_of_domain = 0

    # Check for '@'
    if "@" in msg_id:
        ascii_boundary_char = ord('@')  # The boundary char is '@'
        parts = msg_id.split("@", 1)
        id_part = parts[0]
        domain_part = parts[1]
        # If there's something after '@', we say domain exists
        existence_of_domain = 1 if domain_part else 0
    else:
        # No '@', so the entire msg_id is the ID part
        id_part = msg_id
        domain_part = ""

    # --- ID PART ---
    # Clean the ID part for hex/decimal checks
    id_part_clean = clean_string_for_check(id_part, allow_dot=True)

    # is ID part hex? ignoring '.' if you wish
    is_id_part_hex = 1 if is_hex(id_part_clean.replace(".", "")) else 0
    # is ID part decimal?
    is_id_part_decimal = 1 if is_decimal(id_part_clean.replace(".", "")) else 0

    # Existence of '.' in ID part
    existence_of_dots_in_id_part = 1 if '.' in id_part else 0

    # Existence of special chars in ID part (anything not in [a-zA-Z0-9.])
    special_id = "".join(ch for ch in id_part if ch not in (string.ascii_letters + string.digits + "."))
    existence_of_special_chars_in_id_part = 1 if special_id else 0

    # --- DOMAIN PART ---
    domain_part_clean = clean_string_for_check(domain_part, allow_dot=True)

    # is domain part hex? ignoring '.'
    is_domain_part_hex = 1 if is_hex(domain_part_clean.replace(".", "")) else 0
    # is domain part decimal?
    is_domain_part_decimal = 1 if is_decimal(domain_part_clean.replace(".", "")) else 0

    # Existence of '.' in domain part
    existence_of_dots_in_domain_part = 1 if '.' in domain_part else 0

    # Existence of special chars other than '.'
    special_domain = "".join(ch for ch in domain_part if ch not in (string.ascii_letters + string.digits + "."))
    existence_of_special_chars_in_domain_part = 1 if special_domain else 0
    return {
        "length_message_id" : len(msg_id),
        "ascii_boundary_char": ascii_boundary_char,
        "existence_of_domain": existence_of_domain,
        "is_id_part_hex": is_id_part_hex,
        "is_id_part_decimal": is_id_part_decimal,
        "existence_of_dots_in_id_part": existence_of_dots_in_id_part,
        "existence_of_special_chars_in_id_part": existence_of_special_chars_in_id_part,
        "is_domain_part_hex": is_domain_part_hex,
        "is_domain_part_decimal": is_domain_part_decimal,
        "existence_of_dots_in_domain_part": existence_of_dots_in_domain_part,
        "existence_of_special_chars_in_domain_part": existence_of_special_chars_in_domain_part
    }

def safe_get_all(msg, header):
    """ Récupère un en-tête et retourne une liste de strings ou None si erreur. """
    try:
        values = msg.get_all(header, [])
        return values
    
    except Exception as e:  # Si `msg.get_all(header)` génère une erreur
        print(f"⚠️ Erreur lors de la récupération de `{header}`: {e}")
        return []


def extract_email_thread(msg: EmailMessage) -> dict:
    """
    Extract the Mail address and Domain category features.
    """
    unique_to = set()
    unique_reply_to = set()
    all_addresses = set()
    unique_domains = set()
    unique_to_domains = set()
    return_path = ""

    # Extract From, To, Reply-To, and other fields from the email structure
    from_field = msg.get_all("From", [])
    to_field = safe_get_all(msg,"To")
    reply_to_field = msg.get_all("Reply-To", [])
    cc_field = msg.get_all("Cc", [])
    bcc_field = msg.get_all("Bcc", []) 
    return_path_field = msg.get_all("Return-Path", [])
    # Extract domain related features
    from_domain = extract_domain(from_field)
    reply_to_domain=extract_domain(reply_to_field)
    to_domains = extract_domain(to_field)

    all_addresses.update(set(extract_addresses(from_field)))
    all_addresses.update(set(extract_addresses(to_field)))
    all_addresses.update(set(extract_addresses(reply_to_field)))
    all_addresses.update(set(extract_addresses(cc_field)))
    all_addresses.update(set(extract_addresses(bcc_field)))
    all_addresses.update(set(extract_domain(return_path)))
    unique_domains.update(extract_domain(all_addresses))

    unique_to.update(extract_addresses(to_field))
    unique_to_domains.update(extract_domain(extract_addresses(to_field)))

    unique_reply_to.update(extract_addresses(reply_to_field))
 

    
    return_path = return_path_field
    return_path_domain = extract_domain(return_path)

    num_ofTo_domains_identical_to_From_domain=0
    num_ofCc_domains_identical_to_From_domain = 0
    num_ofBcc_domains_identical_to_From_domain=0

    for from_domain in from_domain:
        for to_domain in to_domains:
            if from_domain==to_domain:
                num_ofTo_domains_identical_to_From_domain+=1
        for cc_domain in extract_domain(cc_field):
            if cc_domain==from_domain:
                num_ofCc_domains_identical_to_From_domain +=1
        for bcc_domain in extract_domain(bcc_field):
            if bcc_domain==from_domain:
                num_ofBcc_domains_identical_to_From_domain+=1

    return {
    'num_From_header_tags_in_thread' : len(msg.get_all("From",[])),
    'num_ofToheader_tags_in_thread' : len(to_field),
    'num_ofunique_To_addresses_in_thread' : len(unique_to),
    'num_ofunique_Reply-To_addresses_in_thread' : len(unique_reply_to),
    'num_ofuniquedomains_in_all_email_addresses' : len(unique_domains),

    'ratio_unique_To_domains_to_unique_domains_in_all_addresses': len(unique_to_domains) / len(unique_domains) if unique_domains else 0,

    'num_Cc_addresses' : len(set(extract_addresses(cc_field))),
    'num_Bcc_addresses' : len(set(extract_addresses(bcc_field))),
    'num_of_Sender_addresses' : len(extract_addresses(msg.get_all("Sender",[]))),

    'IsReturn_Path_identical_to_Reply_To' : int(
    bool(set(extract_domain(return_path)) & set(extract_domain(reply_to_field)))),

    'num_ofTo_domains_identical_to_From_domain' : num_ofTo_domains_identical_to_From_domain,


    'num_ofCc_domains_identical_to_From_domain': num_ofCc_domains_identical_to_From_domain,

    'num_ofBcc_domains_identical_to_From_domain': num_ofBcc_domains_identical_to_From_domain,

    'IsFrom_domain_same_as_Reply-To_domain': 1 if (len(from_domain)!=0 and len(reply_to_domain)!=0) and from_domain[0] == reply_to_domain[0] else 0,
    'IsFrom_domain_same_as_Return-Path_domain': 1 if (len(from_domain)!=0 and len(return_path_domain)!=0) and from_domain[0] == return_path_domain[0] else 0,


    'IsFrom_entity_bracketed': 1 if len(from_field)!=0 and any(
        re.search(r'<[^>]+>', field) for field in [from_field[0]]) else 0,

    'IsTo_entity_bracketed': 1 if any(
    re.search(r'<[^>]+>', field) for field in to_field) else 0
    }


def extract_addresses(header_values: list[str]) -> list[str]:
    """
    Given a list of strings, extract all email addresses using regex.
    """
    emails = []
    for value in header_values:
        emails.extend(re.findall(r'[\w\.-]+@[\w\.-]+', value))
    return emails

    
def extract_emails_from_text(text):
    """ Extrait toutes les adresses email d'un texte """
    EMAIL_REGEX = r"[\w\.-]+@[\w\.-]+\.\w+"
    return re.findall(EMAIL_REGEX, text)



def extract_domain(email_address: list[str]) -> list[str]:
    """
    Extract the domain from an email address.
    """
    domains=[]
    for address in email_address:
        if '@' in address:
            domains.append(address.split('@')[-1])
    return domains

def extract_section_boundary(msg : EmailMessage)->dict:
    boundary = msg.get_boundary()

    if boundary:

        return {
            'len_first_boundary' : len(boundary),
            'boundary_start_with_=' : 1 if boundary.startswith("=") else 0,
            'exist_=_middle_boundary' : 1 if "=" in boundary[1:-1] else 0,
            'exist_underscore_boundary' : 1 if "_" in boundary else 0,
            'exist_dot_boundary' : 1 if "." in boundary else 0,
            'exist_other_special_char_boundary' : 1 if bool(re.search(r'[^a-zA-Z0-9=_.-]', boundary)) else 0,
            'boundary_hexadecimal' : 1 if all(c in '0123456789abcdefABCDEF=_.-' for c in boundary) else 0,
            'boundary_decimal' : 1 if boundary.replace('=','').replace('_','').replace('.','').isdigit() else 0
                }
    
    else : 
        return {
            'len_first_boundary' : 0,
            'boundary_start_with_=' : 0,
            'exist_=_middle_boundary' : 0,
            'exist_underscore_boundary' : 0,
            'exist_dot_boundary' : 0,
            'exist_other_special_char_boundary' : 0,
            'boundary_hexadecimal' : 0,
            'boundary_decimal' : 0
                }

def extraction_character_set(msg: EmailMessage) -> dict:
    unique_charsets = set()
    first_charset_index = -1  # Index du charset dans la première section trouvée
    first_section_found = False

    for part in msg.walk():
        content_type = part.get_content_type()
        content_params = part.get_params() 

        # Extraction du charset si présent
        charset = part.get_content_charset()
        if charset:
            unique_charsets.add(charset.lower())

        # Vérifier si c'est la première section et qu'un charset y est défini
        if not first_section_found and content_type.startswith("text/") and content_params:
            first_section_found = True  # Marquer qu'on a trouvé la première section
            # Rechercher "charset=" dans les paramètres
            for i, (param_name, param_value) in enumerate(content_params):
                if param_name.lower() == "charset":
                    first_charset_index = i
                    break 

    return {
        'index_charset_first_sec': first_charset_index,  # Index du charset dans la première section
        'num_unique_charset': len(unique_charsets)  # Nombre de charsets uniques
    }


def extract_css_JS(html_contents: list[str]) -> dict:
    total_style_length = 0
    rtl_exists = False
    js_exists = False

    for html in html_contents:
        if not html or len(html.strip()) == 0:
            continue

        # <style> ... </style>
        style_bodies = re.findall(r'<style[^>]*>(.*?)</style>', html, re.DOTALL | re.IGNORECASE)
        total_style_length += sum(len(style) for style in style_bodies)

        # style="..." inline
        inline_styles = re.findall(r'style=["\']([^"\']*)["\']', html, re.IGNORECASE)
        total_style_length += sum(len(style) for style in inline_styles)

        # Check for direction: rtl
        if not rtl_exists and re.search(r'direction\s*:\s*rtl', html, re.IGNORECASE):
            rtl_exists = True

        # Check for <script>
        if not js_exists and re.search(r'<script[^>]*>(.*?)</script>', html, re.IGNORECASE | re.DOTALL):
            js_exists = True

    return {
        'length_style': total_style_length,
        'exist_direction': int(rtl_exists),
        'exist_JS': int(js_exists)
    }



def check_inline_script_presence(html_content: str) -> int:
    """
    Vérifie la présence de balises <script> dans le contenu HTML.
    Renvoie 1 si au moins une balise <script> est détectée, sinon 0.
    """
    # Utilise une expression régulière pour détecter les balises <script> en ligne
    if re.search(r'<script[^>]*>(.*?)</script>', html_content, re.IGNORECASE | re.DOTALL):
        return 1
    return 0


def extract_dom_features(html_content:list[str]) -> dict:
    """
    Extrait les caractéristiques de la structure DOM d'un e-mail HTML.
    Renvoie un dictionnaire avec la profondeur, le nombre de nœuds feuilles,
    les types uniques de nœuds feuilles, la profondeur moyenne et l'écart type.
    """
    dom_features = {
        'depht_DOM_tree': 0,
        'num_DOM_leaf': 0,
        'num_unique_DOM_leaf': 0,
        'mean_DOM_leaf_depths': 0.0,
        'standart_dev_DOM_depths': 0.0
    }

    longest_html = max(html_content, key=len)

    
    soup = BeautifulSoup(longest_html, 'html.parser')

    # Stocker les profondeurs des nœuds feuilles
    leaf_node_depths = []
    unique_leaf_node_types = set()

    # Commencer le parcours depuis la balise <html> ou la racine du document
    root = soup.find('html') or soup
    traverse_dom_tree(root, leaf_node_depths, unique_leaf_node_types, dom_features, depth=0)

    # Calculer les métriques des nœuds feuilles
    dom_features['num_DOM_leaf'] = len(leaf_node_depths)
    dom_features['num_unique_DOM_leaf'] = len(unique_leaf_node_types)

    if leaf_node_depths:
        dom_features['mean_DOM_leaf_depths'] = np.mean(leaf_node_depths)
        dom_features['standart_dev_DOM_depths'] = np.std(leaf_node_depths)

    return dom_features


def traverse_dom_tree(node, leaf_node_depths, unique_leaf_node_types, dom_features, depth=0):
    """
    Parcours récursif de l'arbre DOM pour extraire les caractéristiques liées aux nœuds feuilles.
    """
    if hasattr(node, 'name') and node.name:
        # Mettre à jour la profondeur maximale de l'arbre DOM
        dom_features['depht_DOM_tree'] = max(dom_features['depht_DOM_tree'], depth)

        # Si le nœud est une feuille (n'a pas d'enfant de type balise HTML)
        if not node.find_all(True):
            leaf_node_depths.append(depth)
            unique_leaf_node_types.add(node.name)

        # Parcourir les enfants du nœud actuel
        for child in node.children:
            traverse_dom_tree(child, leaf_node_depths, unique_leaf_node_types, dom_features, depth + 1)

def extract_link_features(html_contents: list[str], index) -> tuple[dict, list[str]]:
    """
    Extrait les fonctionnalités liées aux adresses e-mail et aux URLs dans toutes les sections text/html d'un e-mail.
    Renvoie un dictionnaire avec les stats globales + toutes les URLs cliquables.
    """
    email_url_features = {
        'num_email_addresses_in_html_section': 0,
        'num_a_tags': 0,
        'num_URLs_in_html_section': 0,
        'ratio_unique_domains_to_domains_of_allURLs_in_anytags': 0.0
    }

    all_domains = set()
    all_urls = []
    clickable_urls_total = []

    for html_content in html_contents:
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. Adresses e-mail
        email_addresses = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', html_content)
        email_url_features['num_email_addresses_in_html_section'] += len(email_addresses)

        # 2. Liens <a>
        a_tags = soup.find_all('a', href=True)
        saferedirect_tags = soup.find_all('a', {'data-saferedirecturl': True})
        email_url_features['num_a_tags'] += len(a_tags) + len(saferedirect_tags)

        # 3. URLs
        clickable_urls = re.findall(r'href=["\'](https?://[^\s"\'<>]+)["\']', html_content)
        urls = re.findall(r'(https?://[^\s"\'<>]+)["\']', html_content)

        clickable_urls_total.extend(clickable_urls)
        all_urls.extend(urls)

        # Extraire les domaines pour le ratio
        for url in urls:
            domain = extract_domain_from_url(url, all_domains)
            if domain:
                all_domains.add(domain)

    # 4. Statistiques finales
    email_url_features['num_URLs_in_html_section'] = len(all_urls)
    total_domains = len(all_urls)
    unique_domains = len(all_domains)

    if total_domains > 0:
        email_url_features['ratio_unique_domains_to_domains_of_allURLs_in_anytags'] = unique_domains / total_domains

    return email_url_features, clickable_urls_total



def extract_domain_from_url(url: str,setDomains : set) -> str:
    """
    Extrait le domaine à partir d'une URL donnée.
    """
    try:
        if "CLICK_THROUGH" not in url:
            parsed_url = urlparse(url)
            setDomains.add(parsed_url.netloc)
            return parsed_url.netloc.lower()
        return "CLICK_THROUGH"
    except Exception as e:
        print(f"Erreur lors de l'extraction du domaine : {e}")
        return ""


def extract_urls_from_text(text, emails):
    URL_REGEX = r"(https?://[^\s]+|www\.[^\s]+)"
    """ Extrait toutes les URLs du texte, en excluant les domaines des emails """
    urls = re.findall(URL_REGEX, text)
    
    # Exclure les domaines des emails détectés
    email_domains = extract_domain(emails)  # Extraire uniquement les domaines
    filtered_urls =[]
    for url in urls:
        if url not in email_domains :
            filtered_urls.append(url)
        else :
            email_domains.remove(url)
    return filtered_urls


def normalize_url(url):
    """ Normalise une URL en imposant https:// et en supprimant les fragments """
    parsed_url = urlparse(url)

    # Si l'URL ne contient pas de schéma (http, https), ajouter https:// par défaut
    scheme = parsed_url.scheme if parsed_url.scheme else "https"
    normalized_url = urlunparse((scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
    
    return normalized_url.lower()  # Convertir en minuscule pour éviter la casse


def extract_structure_features(msg: EmailMessage, index: int) -> dict:
    
    features = {}
    html_content = []
    text_content = ""
    
    # Variables pour stocker les statistiques
    num_text_plain = 0
    num_text_html = 0
    num_image_sections = 0
    num_application_sections = 0
    length_text_html = 0
    charsets = set()
    first_charset_index = -1
    first_section_found = False
    total_urls_text_plain_and_html = set()
    clickable_urls_html = []
    
    # *** SINGLE PASS SUR L'EMAIL ***

    
    for part in msg.walk():
        ctype = part.get_content_type().lower()
        charset = part.get_content_charset()
        encoding = part.get('Content-Transfer-Encoding', '').lower()

        if charset:
            charsets.add(charset.lower())

        if charset is None or charset.lower() == "default_charset" or charset.lower() == "default" or charset.lower()=="chinesebig5" or charset.lower()=="unknown-8bit" or charset.lower()=="gb2312_charset" or charset.lower()=="iso-4470lgm1879-728jaagth":
            charset = "utf-8"
        # Récupérer le charset de la première section texte rencontrée
        if not first_section_found and ctype.startswith("text/") and charset!=None:
            first_section_found = True
            params = part.get_params()
            if params!=None:
                for i, (param_name, param_value) in enumerate(params):
                    if param_name.lower() == "charset":
                        first_charset_index = i
                        break
        # Extraire le contenu
        if ctype == "text/plain":
            num_text_plain += 1
            try:
        # Récupérer l'encodage

                encoding = part.get('Content-Transfer-Encoding', '').lower()

               

                # Obtenir le contenu brut encodé
                raw_payload = part.get_payload(decode=False)

                # Décodage selon l'encodage
                if encoding == 'base64':
                    decoded_bytes = base64.b64decode(raw_payload)
                elif encoding == 'quoted-printable':
                    decoded_bytes = quopri.decodestring(raw_payload)
                else:
                    decoded_bytes = part.get_payload(decode=True)  # fallback

                # Convertir les bytes en texte
                text_content_intermediaire = decoded_bytes.decode(charset, errors='replace')

                # Vérifie s’il y a du HTML dans le text/plain
                if "<html" in text_content_intermediaire.lower() or "<a " in text_content_intermediaire.lower():
                    soup = BeautifulSoup(text_content_intermediaire, "html.parser")
                    text_content_intermediaire = soup.get_text(separator="\n", strip=True)

                # Ajoute avec un saut de ligne
                if text_content:
                    text_content += '\n'
                text_content += text_content_intermediaire
            except Exception as e:
                print(f"[ERREUR] Impossible de décoder text/plain: {e }")
                print(index)
                text_content = "-1"

        # 🔹 Si `text_content` est vide après extraction, le remplacer par `-1`
        
        elif ctype == "text/html":
            num_text_html += 1
            try:
                html_content.append(part.get_payload(decode=True).decode(charset or 'utf-8', errors='replace'))
            except:
                html_content = ""

        elif ctype.startswith("image/"):
            num_image_sections += 1
        
        elif ctype.startswith("application/"):
            num_application_sections += 1
    if not text_content.strip():
            text_content = "-1"

    # Stockage des statistiques
    features.update({
        'number_of_text_plain_sections': num_text_plain,
        'number_of_text_html_sections': num_text_html,
        'number_of_image_sections': num_image_sections,
        'number_of_application_sections': num_application_sections,
        'ratio_text_plain_to_any_text_sections': num_text_plain / (num_text_plain + num_text_html) if (num_text_plain + num_text_html) > 0 else 0,
        'length_of_text_in_text_html_section': length_text_html,
        'index_charset_first_sec': first_charset_index,
        'num_unique_charset': len(charsets)
    })

    # *** Extraction des features à partir des données collectées ***
    features.update(extract_header_statistics(msg))
    features.update(extract_MIME_version(msg))
    features.update(extract_message_id_features(msg))
    features.update(extract_email_thread(msg))
    features.update(extract_section_boundary(msg))

    # *** Traitement du HTML en une seule fois ***
    if len(html_content)!=0 :
        length_text_html = len(max(html_content, key=len))
        features.update(extract_css_JS(html_content))
        features.update(extract_dom_features(html_content))
        link_features, clickable_urls_html = extract_link_features(html_content, index)
        features.update(link_features)
    else:
        
        features.update({
        'depht_DOM_tree': 0,
        'num_DOM_leaf': 0,
        'num_unique_DOM_leaf': 0,
        'mean_DOM_leaf_depths': 0.0,
        'standart_dev_DOM_depths': 0.0,
        'num_email_addresses_in_html_section': 0,
        'num_a_tags': 0,
        'num_URLs_in_html_section': 0,
        'ratio_unique_domains_to_domains_of_allURLs_in_anytags': 0.0,
        'length_style' : 0,
        'exist_direction' : 0,
        'exist_JS' : 0

    })
    total_urls_text_plain_and_html.update(clickable_urls_html)
    return features, total_urls_text_plain_and_html, text_content

def main():
    csv_path = 'email_perso_liza/perso_liza_without_pub.csv'
    data = pd.read_csv(csv_path)
    emails = data['email']
    labels = data['label']

    data_out = []
    data_urls_out=[]
    data_text_out=[]
    id_email = ""
    start_time = time.time()
    for idx, email in enumerate(emails):
        try:
            msg = load_email_from_string(email)
            features,clickable_urls, text_content = extract_structure_features(msg,idx)

            #si l'email n'est pas de Message-id comme email 5010 dans spam assassin
            try :
                id_email=msg.get("Message-ID")
            except:
                id_email=idx
            #csv feature
            features["label"] = labels.iloc[idx]
            features["email_id"] = id_email
            features["index"] = idx
            data_out.append(features)
            #csv url
            if clickable_urls:
                for url in clickable_urls:
                    data_urls_out.append({
                        "index": idx, "url": url, "label": labels.iloc[idx]
                    })
            else:
                # **Ajout d'un email sans URL avec -1**
                data_urls_out.append({
                    "index": idx, "url": "-1", "label": labels.iloc[idx]
                })

            #csv text
            data_text_out.append({"index": idx, "email_id" : id_email,"text": text_content, "label" : labels.iloc[idx]})
        except Exception as e:
            print(f"Erreur lors du traitement de l'email à l'index {idx}: {e}")
    end_time = time.time()  # Capture le temps de fin
    execution_time = end_time - start_time  # Temps écoulé
    print(f"Temps d'exécution : {execution_time:.6f} secondes")

    df_features = pd.DataFrame(data_out)
    df_features.to_csv("email_perso_liza/features_perso_liza_without_pub.csv", index=False, encoding='utf-8')

    urls_extracted=pd.DataFrame(data_urls_out)
    #urls_extracted.to_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_urls/urls_train_set_20.csv", index=False, encoding='utf-8')

    text_extracted = pd.DataFrame(data_text_out)
    #text_extracted.to_csv("enron_spamassassin_nazario_nigerian_best_compo/extracted_texts/text_train_set_20.csv", index=False, encoding='utf-8')
    
    print("Les CSV étendus ont été créés avec succès")

if __name__ == "__main__":
    main() 