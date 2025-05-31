import sys
import json
import datetime

from ns_log import NsLog
from json2arff import json2arff
from traceback import format_exc
from domain_parser import domain_parser
from rule_extraction import rule_extraction


class Train:
    def __init__(self):
        self.logger = NsLog("log")

        self.json2arff_object = json2arff()
        self.parser_object = domain_parser()
        self.rule_calculation = rule_extraction()

        self.path_input = "urlsModule/nlp_features/nlp_features_extraction/input/"
        self.path_arff = "urlsModule/nlp_features/nlp_features_extraction/output/arff/"
        self.path_features = "urlsModule/nlp_features/nlp_features_extraction/output/features/"
        self.path_parsed_domain = "urlsModule/nlp_features/nlp_features_extraction/output/domain_parser/"

    def domain_parser(self, param):
        parsed_domains = []

        for i in range(1, len(param), 2):
            try:
                if param[i + 1] in ('phish', 'legitimate'):
                    with open(f"{self.path_input}{param[i]}", "r", encoding="utf-8") as f:
                        dataset = json.load(f)

                    parsed = self.parser_object.parse(dataset, param[i + 1], len(parsed_domains))
                    for entry, raw in zip(parsed, dataset):
                        entry["id"] = raw.get("index", -1)
                    parsed_domains += parsed
                else:
                    self.logger.debug("class labels must be entered one of (phish, legitimate)")

            except Exception as e:
                self.logger.error(f"an error occurred: {format_exc()}")
                self.logger.debug(f"Error when processing file: {param[i]}")

        self.logger.info(f" {len(parsed_domains)} URLs parsed")
        return parsed_domains

    def json_to_file(self, name, path, data):
        if name == "features":
            file_name = "extraction_features.txt"
        else:
            time_now = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
            file_name = name + "_" + time_now + ".txt"
        
        with open(path + file_name, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        
        self.logger.info(f"{name} written to file: {file_name}")


    def arff_to_file(self, name, path, data):
        time_now = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        file_name = name + "_" + time_now + ".txt"
        with open(path + file_name, "w", encoding="utf-8") as file:
            file.write(data)
        self.logger.info(f"{name} written to file.")


def main():
    tr_obj = Train()
    parsed_domains = tr_obj.domain_parser(sys.argv)
    #tr_obj.json_to_file("parse", tr_obj.path_parsed_domain, parsed_domains)

    features = tr_obj.rule_calculation.extraction(parsed_domains)
    tr_obj.json_to_file("features", tr_obj.path_features, features)

    #arff_str = tr_obj.json2arff_object.convert_for_train(features, '')
    #tr_obj.arff_to_file("arff", tr_obj.path_arff, arff_str)


if __name__ == "__main__":
    main()
