import os
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

class SurefireReportParser:
    """
    A class to parse Surefire XML reports and extract testsuite results
    based on the testsuite name (which usually corresponds to the
    fully qualified class name).
    """
    def __init__(self, reports_dir="target/surefire-reports"):
        """
        Initializes the SurefireReportParser with the directory containing the reports.

        Args:
            reports_dir: The directory containing the Surefire XML reports.
                         Defaults to "target/surefire-reports".
        """
        self.reports_dir = reports_dir

    def get_testsuite_results_by_name(self, testsuite_name):
        """
        Retrieves the testsuite results from the Surefire XML report
        for a specific testsuite name (usually the fully qualified class name).

        Args:
            testsuite_name: The fully qualified name of the testsuite
                            (e.g., "csv.converter.IgnoreMissingValuesConverterTest").

        Returns:
            A dictionary containing 'tests', 'failures', 'errors', 'skipped', and 'time'
            for the specified testsuite. Returns None if the report file is not found
            or cannot be parsed.
        """
        logger.info(testsuite_name)
        filename = f"TEST-{testsuite_name}.xml"
        report_path = os.path.join(self.reports_dir, filename)

        if not os.path.exists(report_path):
            logger.warning(f"Report file not found at {report_path}")
            return None

        try:
            tree = ET.parse(report_path)
            root = tree.getroot()
            return {
                'tests': int(root.get('tests', 0)),
                'failures': int(root.get('failures', 0)),
                'errors': int(root.get('errors', 0)),
                'skipped': int(root.get('skipped', 0)),
                'time': float(root.get('time', 0.0))
            }
        except ET.ParseError:
            logger.error(f"Could not parse XML file at {report_path}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing {report_path}: {e}")
            return None


if __name__ == "__main__":
    # Example usage:
    parser = SurefireReportParser()
    testsuite = "csv.converter.IgnoreMissingValuesConverterTest"
    testsuite_results = parser.get_testsuite_results_by_name(testsuite)

    if testsuite_results:
        logger.info(f"Testsuite Results for {testsuite}:")
        logger.info(testsuite_results)
        logger.info(f"Tests: {testsuite_results['tests']}")
        logger.info(f"Failures: {testsuite_results['failures']}")
        logger.info(f"Errors: {testsuite_results['errors']}")
        logger.info(f"Skipped: {testsuite_results['skipped']}")
        logger.info(f"Time: {testsuite_results['time']}")
    else:
        logger.error(f"Could not retrieve testsuite results for {testsuite}")