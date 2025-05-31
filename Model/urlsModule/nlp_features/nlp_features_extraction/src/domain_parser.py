import tldextract
import re
from ns_log import NsLog
from tqdm import tqdm

class domain_parser(object):

    def __init__(self):
        self.logger = NsLog("log")

    def parse(self, domain_list, class_info, count):
        self.logger.info("domain_parser.parse() is running")

        parsed_domain_list = []
        registered_domain_lst = []

        for item in tqdm(domain_list):
            if isinstance(item, dict) and "url" in item:
                url = item["url"].strip()
                index = item.get("index", count)
            elif isinstance(item, str):
                url = item.strip()
                index = count
            else:
                continue

            extracted_domain = tldextract.extract(url)
            registered_domain_lst.append(extracted_domain.registered_domain)

            domain = {
                "url": url,
                "domain": extracted_domain.domain,
                "registered_domain": extracted_domain.registered_domain,
                "tld": extracted_domain.suffix,
                "subdomain": extracted_domain.subdomain,
                "class": class_info,
                "id": index,
                "protocol": url.split("://")[0] if "://" in url else '',
            }

            tmp = url[url.find(extracted_domain.suffix):]
            pth = tmp.partition("/")
            domain["path"] = pth[1] + pth[2]

            domain["words_raw"] = self.words_raw_extraction(
                extracted_domain.domain, extracted_domain.subdomain, pth[2]
            )

            parsed_domain_list.append(domain)
            count += 1

        return parsed_domain_list

    def parse_nonlabeled_samples(self, domain_list, count=0):
        self.logger.info("domain_parser.parse_nonlabeled_samples() is running")
        parsed_domain_list = []
        registered_domain_lst = []

        for item in tqdm(domain_list):
            if isinstance(item, dict) and "url" in item:
                url = item["url"].strip()
                index = item.get("index", count)
            elif isinstance(item, str):
                url = item.strip()
                index = count
            else:
                continue

            extracted_domain = tldextract.extract(url)
            registered_domain_lst.append(extracted_domain.registered_domain)

            domain = {
                "url": url,
                "domain": extracted_domain.domain,
                "registered_domain": extracted_domain.registered_domain,
                "tld": extracted_domain.suffix,
                "subdomain": extracted_domain.subdomain,
                "id": index,
                "protocol": url.split("://")[0] if "://" in url else '',
            }

            tmp = url[url.find(extracted_domain.suffix):]
            pth = tmp.partition("/")
            domain["path"] = pth[1] + pth[2]

            domain["words_raw"] = self.words_raw_extraction(
                extracted_domain.domain, extracted_domain.subdomain, pth[2]
            )

            parsed_domain_list.append(domain)
            count += 1

        return parsed_domain_list

    def words_raw_extraction(self, domain, subdomain, path):
        w_domain = re.split(r"[-./?=:@&%:_]", domain.lower())
        w_subdomain = re.split(r"[-./?=:@&%:_]", subdomain.lower())
        w_path = re.split(r"[-./?=:@&%:_]", path.lower())

        raw_words = w_domain + w_path + w_subdomain
        return list(filter(None, raw_words))
