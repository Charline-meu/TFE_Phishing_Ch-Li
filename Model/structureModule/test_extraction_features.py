import unittest
from email import message_from_string
from extraction_features import extract_structure_features, load_email_from_string
import pandas as pd
class TestEmailFeatureExtraction(unittest.TestCase):

    def setUp(self):
        # Example email to test
        df = pd.read_csv('dataset/spam_assassin_original.csv')
        self.test_email = df.loc[61, 'email']

        self.msg = load_email_from_string(self.test_email)

    def test_extract_structure_features(self):
        # Extract features
        features_structure,extracted_urls,text = extract_structure_features(self.msg, index=0)

        # Expected values
        expected_features_structure = {
            'number_of_text_plain_sections': 1,
            'number_of_text_html_sections': 1,
            'number_of_image_sections':0,
            'number_of_application_sections' : 0,
            'ratio_text_plain_to_any_text_sections':0.5,
            'length_of_text_in_text_html_section' :1476,

            'num_standart_header' : 7,
            'num_received': 4, 
            'num_in_reply_to' : 0,
            'exist_x_mailer': 1, 
            'exist_user_agent' : 0,

            'mime_version': 1,
            'case_var_MV' : 0,

            'length_message_id' : 41,
            "ascii_boundary_char": 64,
            "existence_of_domain": 1,
            "is_id_part_hex": 1,
            "is_id_part_decimal": 0,
            "existence_of_dots_in_id_part": 0,
            "existence_of_special_chars_in_id_part": 1,
            "is_domain_part_hex": 0,
            "is_domain_part_decimal": 0,
            "existence_of_dots_in_domain_part": 1,
            "existence_of_special_chars_in_domain_part": 0,

            # le cas avec plusieurs addresse mail en cc ou bcc fonctionne
            'num_From_header_tags_in_thread' : 1,
            'num_ofToheader_tags_in_thread' : 1,
            'num_ofunique_To_addresses_in_thread' : 1,
            'num_ofunique_Reply-To_addresses_in_thread' : 1,
            'num_ofuniquedomains_in_all_email_addresses' : 2,
            'ratio_unique_To_domains_to_unique_domains_in_all_addresses': 1/2,
            'num_Cc_addresses' : 0,
            'num_Bcc_addresses' : 0,
            'num_of_Sender_addresses' : 1,
            'IsReturn_Path_identical_to_Reply_To' : 1,
            'num_ofTo_domains_identical_to_From_domain' : 0,
            'num_ofCc_domains_identical_to_From_domain': 0,
            'num_ofBcc_domains_identical_to_From_domain': 0,
            'IsFrom_domain_same_as_Reply-To_domain': 0,
            'IsFrom_domain_same_as_Return-Path_domain': 0,
            'IsFrom_entity_bracketed': 1,
            'IsTo_entity_bracketed': 0,

            'len_first_boundary' : 41,
            'boundary_start_with_=' : 0,
            'exist_=_middle_boundary' : 1,
            'exist_underscore_boundary' : 1,
            'exist_dot_boundary' : 1,
            'exist_other_special_char_boundary' : 0,
            'boundary_hexadecimal' : 0,
            'boundary_decimal' : 0,

            #le cas avec un autre index de charser fonctione et différent charset
            'index_charset_first_sec' : 1,
            'num_unique_charset' : 1,

            'depht_DOM_tree': 4,
            'num_DOM_leaf': 24,
            'num_unique_DOM_leaf': 6,
            'mean_DOM_leaf_depths': 3.2916666666666665,
            'standart_dev_DOM_depths': 0.7347996703561832,

            #testé en rajoutant manuellement dans le mail, ça fonctionne
            'length_style' : 0,
            'exist_direction' :0,
            'exist_JS' : 0,

            #pareil j'ai rajouté des urls,.... ça fonctionne
            'num_email_addresses_in_html_section': 2,
            'num_a_tags': 2,
            'num_URLs_in_html_section': 1,
            'ratio_unique_domains_to_domains_of_allURLs_in_anytags': 1.0



        }
                # Check feature extraction
        for key, expected_value in expected_features_structure.items():
            with self.subTest(feature=key):
                self.assertEqual(features_structure.get(key, None), expected_value)

if __name__ == '__main__':
    unittest.main()
