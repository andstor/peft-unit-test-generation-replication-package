import re
import xml.etree.ElementTree as ET
import pandas as pd

class PITesterReport:
    def __init__(self, xml_string: str):
        self.root = ET.fromstring(xml_string)
        self.df = self.to_dataframe()
        # normalize test names for filtering
        
    @staticmethod
    def normalize_test_id(full_test_id: str) -> str:
        """
        Convert PIT test identifier to a normalized form:
        - Keep package and class
        - Keep method name
        - Remove parameterized index [..] and trailing parentheses
        Examples:
        - com.humaneval.SolutionJava0Test.testJava1(com.humaneval.SolutionJava0Test) -> com.humaneval.SolutionJava0Test.testJava1
        - com.humaneval.SolutionJava1Test.testJave[1](com.humaneval.SolutionJava1Test) -> com.humaneval.SolutionJava1Test.testJave
        """
        # Remove parameterized index [..] before the parentheses
        without_index = re.sub(r"\[\d+\]", "", full_test_id)
        # Remove parentheses with class at the end
        normalized = re.sub(r"\(.*\)$", "", without_index)
        return normalized

    def to_records(self):
        records = []
        for mutation in self.root.findall("mutation"):
            killing = mutation.findtext("killingTests") or ""
            succeeding = mutation.findtext("succeedingTests") or ""
            record = {
                "detected": mutation.attrib.get("detected") == "true",
                "status": mutation.attrib.get("status"),
                "numberOfTestsRun": int(mutation.attrib.get("numberOfTestsRun", 0)),
                "sourceFile": mutation.findtext("sourceFile"),
                "mutatedClass": mutation.findtext("mutatedClass"),
                "mutatedMethod": mutation.findtext("mutatedMethod"),
                "methodDescription": mutation.findtext("methodDescription"),
                "lineNumber": int(mutation.findtext("lineNumber")),
                "mutator": mutation.findtext("mutator"),
                "indexes": [i.text for i in mutation.find("indexes").findall("index")] if mutation.find("indexes") is not None else [],
                "blocks": [b.text for b in mutation.find("blocks").findall("block")] if mutation.find("blocks") is not None else [],
                "killingTests": killing.split("|") if killing else [],
                "succeedingTests": succeeding.split("|") if succeeding else [],
                "description": mutation.findtext("description"),
            }
            records.append(record)
        return records

    def to_dataframe(self):
        if not hasattr(self, 'df'):
            self.df = pd.DataFrame(self.to_records())
        return self.df

    def get_mutations_by_test(self, test_identifier: str, status_filter: str):
        """
        Returns mutations affected by a specific test.
        status_filter: "KILLED", "SURVIVED", "NO_COVERAGE"
        The test_identifier should be in normalized form (package.class.method).
        """
        def normalize_list(tests):
            return [self.normalize_test_id(t) for t in tests]

        if status_filter == "KILLED":
            return self.df[self.df["killingTests"].apply(lambda tests: test_identifier in normalize_list(tests))]
        elif status_filter == "SURVIVED":
            return self.df[self.df["succeedingTests"].apply(lambda tests: test_identifier in normalize_list(tests))]
        elif status_filter == "NO_COVERAGE":
            return self.df[self.df["status"] == "NO_COVERAGE"]
        else:
            raise ValueError("status_filter must be 'KILLED', 'SURVIVED', or 'NO_COVERAGE'")

# Example usage
if __name__ == "__main__":
    with open("target/pit-reports/mutations.xml", "r", encoding="utf-8") as f:
        xml_content = f.read()

    report = PITesterReport(xml_content)

    test_identifier = "com.humaneval.SolutionJava0Test.testJava1(com.humaneval.SolutionJava0Test)"
    killed_by_test = report.get_mutations_by_test(test_identifier, "KILLED")
    survived_by_test = report.get_mutations_by_test(test_identifier, "SURVIVED")
    no_coverage = report.get_mutations_by_test(test_identifier, "NO_COVERAGE")

    print("KILLED by test:", killed_by_test.shape[0])
    print("SURVIVED by test:", survived_by_test.shape[0])
    print("NO_COVERAGE:", no_coverage.shape[0])