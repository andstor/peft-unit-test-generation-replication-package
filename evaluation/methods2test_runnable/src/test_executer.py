from typing import Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path
import subprocess
import pandas as pd
import xml.etree.ElementTree as ET
import abc
import os
from .java_utils import extract_java_info

import logging
logger = logging.getLogger(__name__)


@dataclass
class TestCandidateDescriptor:
    function_identifier: Dict
    class_identifier: Dict
    file: str
    
@dataclass
class FocalMethodDescriptor:
    function_identifier: Dict
    class_identifier: Dict
    signature: str
    file: str



class MavenPlugin(abc.ABC):
    @abc.abstractmethod
    def get_plugin_definition(self) -> Dict:
        raise NotImplementedError


class GradlePlugin(abc.ABC):
    @abc.abstractmethod
    def install(self, build_file: Path):
        raise NotImplementedError


class Tool(abc.ABC):
    def __init__(self, build_working_dir: Path):
        self.build_working_dir = build_working_dir

    @abc.abstractmethod
    def get_report(self, focal_method: FocalMethodDescriptor) -> Optional[Dict]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_report_file(self) -> Path:
        raise NotImplementedError



class MutationTool(Tool):
    def __init__(self, build_working_dir: Path):
        self.build_working_dir = build_working_dir
        
    @abc.abstractmethod
    def get_report(self, focal_method: FocalMethodDescriptor) -> Optional[Dict]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_report_file(self) -> Path:
        raise NotImplementedError

class PITMutationTool(MutationTool):
    
    @abc.abstractmethod
    def get_report_file(self) -> Path:
        raise NotImplementedError

    def get_report(self, focal_method: FocalMethodDescriptor, unit_test: TestCandidateDescriptor) -> Optional[Dict]:
        from .pitester_report import PITesterReport
        mutation_file = self.get_report_file()

        if mutation_file is None:
            logger.warning("No mutation file found")
            return None

        with open(mutation_file) as f:
            
            xml_string = f.read()
            report = PITesterReport(xml_string)
            
            package, _ = extract_java_info(unit_test.file)

            killed_mutants = report.get_mutations_by_test(".".join([package, unit_test.class_identifier, unit_test.function_identifier]), "KILLED")
            killed_mutants = killed_mutants[killed_mutants['mutatedClass'] == ".".join([package, focal_method.class_identifier])]
            killed_mutants = killed_mutants[killed_mutants['mutatedMethod'] == focal_method.function_identifier]

            survived_mutants = report.get_mutations_by_test(".".join([package, unit_test.class_identifier, unit_test.function_identifier]), "SURVIVED")
            survived_mutants = survived_mutants[survived_mutants['mutatedClass'] == ".".join([package, focal_method.class_identifier])]
            survived_mutants = survived_mutants[survived_mutants['mutatedMethod'] == focal_method.function_identifier]

            no_coverage_mutants = report.get_mutations_by_test(None, "NO_COVERAGE")
            no_coverage_mutants = no_coverage_mutants[no_coverage_mutants['mutatedClass'] == ".".join([package, focal_method.class_identifier])]
            no_coverage_mutants = no_coverage_mutants[no_coverage_mutants['mutatedMethod'] == focal_method.function_identifier]

            relevant_mutations = pd.concat([killed_mutants, survived_mutants, no_coverage_mutants])

            mutation_score = len(killed_mutants) / len(relevant_mutations) if len(relevant_mutations) > 0 else 0.0
            return {
                "killed_mutants": len(killed_mutants),
                "survived_mutants": len(survived_mutants),
                "no_coverage_mutants": len(no_coverage_mutants),
                "mutation_score": mutation_score,
            }


class PITMavenMutationTool(PITMutationTool, MavenPlugin):
    
    def get_plugin_definition(self) -> Dict:
        return {
            'groupId': 'org.pitest',
            'artifactId': 'pitest-maven',
            'version': '1.16.0',
            'executions': [
                {
                    'id': 'generate-mutation-report',
                    'phase': 'test',
                    'goals': ['mutationCoverage']
                }
            ],
            'configuration': {
                'outputFormats': { 'param': ['XML'] },
                'fullMutationMatrix': 'true',
            }
        }

    def get_report_file(self) -> Optional[Path]:
        return self.build_working_dir / "target" / "pit-reports" / "mutations.xml"




class CoverageTool(Tool):
    def __init__(self, build_working_dir: Path):
        self.build_working_dir = build_working_dir
    
    @abc.abstractmethod
    def get_report(self, focal_method: FocalMethodDescriptor) -> Optional[Dict]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_report_file(self) -> Path:
        raise NotImplementedError


class JacocoCoverageTool(CoverageTool):
    
    @abc.abstractmethod
    def get_report_file(self) -> Path:
        raise NotImplementedError
    
    def get_report(self, focal_method: FocalMethodDescriptor) -> Optional[Dict]:
        from .java_descriptor_converter import JavaDescriptorConverter
        from .jacoco_report import JaCoCoReport
        coverage_file = self.get_report_file()

        if coverage_file is None:
            logger.warning("No coverage file found")
            return None

        with open(coverage_file) as f:
            converter = JavaDescriptorConverter()
            
            xml_string = f.read()
            report = JaCoCoReport(xml_string)

            descriptor = converter.signature_to_descriptor(focal_method.signature)
            normalized_descriptor = converter.normalize(descriptor)
            package, class_name = extract_java_info(focal_method.file)
            data = report.match_method(package, class_name, focal_method.function_identifier, normalized_descriptor)
        
            if data["INSTRUCTION_COVERED"] + data["INSTRUCTION_MISSED"] == 0:
                instruction_coverage = 0
            else:
                instruction_coverage = data["INSTRUCTION_COVERED"] / (data["INSTRUCTION_MISSED"] + data["INSTRUCTION_COVERED"])
            data["instruction_coverage"] = instruction_coverage
            
            if data["BRANCH_COVERED"] + data["BRANCH_MISSED"] == 0:
                branch_coverage = 0
            else:
                branch_coverage = data["BRANCH_COVERED"] / (data["BRANCH_MISSED"] + data["BRANCH_COVERED"])
            data["branch_coverage"] = branch_coverage
            return data


class JacocoMavenCoverageTool(JacocoCoverageTool, MavenPlugin):
    
    def get_plugin_definition(self) -> Dict:
        return {
            'groupId': 'org.jacoco',
            'artifactId': 'jacoco-maven-plugin',
            'version': '0.8.12',
            'executions': [
                {
                    'goals': ['prepare-agent']
                },
                {
                    'id': 'generate-code-coverage-report',
                    'phase': 'test',
                    'goals': ['report']
                }
            ]
        }

    def get_report_file(self) -> Optional[Path]:
        return self.build_working_dir / "target" / "site" / "jacoco" / "jacoco.xml"



class JacocoGradleCoverageTool(JacocoCoverageTool, GradlePlugin):
    def install(self, build_file: Path):
        with open(build_file, "a") as f:
            f.write("\n\n")
            f.write("test { \n")
            f.write("    jacoco { \n")
            f.write("        enabled = true \n")
            f.write("    } \n")
            f.write("} \n")
            f.write("jacoco { \n")
            f.write("    toolVersion = '0.8.12' \n")
            f.write("} \n")
            f.write("jacocoTestReport { \n")
            f.write("    reports { \n")
            f.write("        xml.required = true \n")
            f.write("        html.required = True # Optional: Generate HTML report as well\n")
            f.write("        csv.required = false\n")
            f.write("        xml.outputLocation = file(\"$buildDir/reports/jacoco/jacoco.xml\")\n")
            f.write("        html.outputLocation = file(\"$buildDir/reports/jacoco/html\")\n")
            f.write("    } \n")
            f.write("} \n")

    def get_report_file(self) -> Optional[Path]:
        return self.build_working_dir / "build" / "reports" / "jacoco" / "jacoco.xml"


class BuildSystem(abc.ABC):
    def __init__(self, build_file: Path, build_working_dir: Path):
        self.build_file = build_file
        self.build_working_dir = build_working_dir
        self.coverage_tool: Optional['CoverageTool'] = None

    @abc.abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError
    
    @abc.abstractmethod
    def execute(self, test_class_identifier: Dict, test_function_identifier: Dict) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def clean(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def install_tool(self, coverage_tool: 'CoverageTool'):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_results(self, test_candidate: TestCandidateDescriptor) -> Optional[Dict]:
        raise NotImplementedError
        


class MavenBuildSystem(BuildSystem):
    def __init__(self, pom_file: Path, build_working_dir: Path):
        super().__init__(pom_file, build_working_dir)
        
    def get_name(self) -> str:
        return "Maven"

    def install_tool(self, coverage_tool: MavenPlugin):
        self.coverage_tool = coverage_tool
        
        pom_file = self._find_super_pom(self.build_file) or self.build_file
        plugin_definition = coverage_tool.get_plugin_definition()
        
        if self._is_plugin_installed(pom_file, plugin_definition['groupId'], plugin_definition['artifactId']):
            self._remove_plugin(pom_file, plugin_definition['groupId'], plugin_definition['artifactId'])
        self._add_plugin(pom_file, plugin_definition)

    
    def execute(self, test_class_identifier: Dict, test_function_identifier: Dict) -> List[str]:
        cmd = ["mvn", "test", "-Dmaven.test.failure.ignore=true", f"-Dtest={test_class_identifier}#{test_function_identifier}"]
        logger.debug(f"Executing command: {' '.join(cmd)}")
        logger.debug(f"Working directory: {self.build_working_dir}")
        logger.debug(f"Test class identifier: {test_class_identifier}")
        logger.debug(f"Test function identifier: {test_function_identifier}")
        logger.debug(f"Build file: {self.build_file}")
        logger.debug(f"Build working directory: {self.build_working_dir}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                text=True,               # Decode output to string
                check=True,               # Raise exception on non-zero exit codes
                cwd=self.build_working_dir,  # Set working directory
                timeout=60
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.CalledProcessError as e:
            return e.stdout, e.stderr, e.returncode

    def clean(self) -> None:
        cmd = ["mvn", "clean"]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                text=True,               # Decode output to string
                check=True,               # Raise exception on non-zero exit codes
                cwd=self.build_working_dir,  # Set working directory
                timeout=60
            )
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return e.stdout, e.stderr, e.returncode

    def get_results(self, test_candidate: TestCandidateDescriptor) -> Optional[Dict]:
        from src.surefire_report import SurefireReportParser
        parser = SurefireReportParser(self.build_working_dir / "target" / "surefire-reports")
        
        package, class_name = extract_java_info(test_candidate.file)
        results = parser.get_testsuite_results_by_name(package + "." + class_name)
        return results

    
    def _find_super_pom(self, pom_file: Path) -> Optional[Path]:
        current_dir = pom_file.parent
        while current_dir != current_dir.parent:  # Stop at the filesystem root
            parent_pom = current_dir / "pom.xml"
            if parent_pom.exists() and self._has_build_section(parent_pom):
                return parent_pom
            current_dir = current_dir.parent
        return None

    def _has_build_section(self, pom_file: Path) -> bool:
        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()
            namespace = root.tag.split('}')[0][1:]
            ET.register_namespace('', namespace)
            namespaces = {'pom': namespace}
            build = root.find('pom:build', namespaces)
            return build is not None
        except ET.ParseError:
            return False

    def _is_plugin_installed(self, pom_file: Path, group_id: str, artifact_id: str) -> bool:
        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()
            namespace = root.tag.split('}')[0][1:]
            ET.register_namespace('', namespace)
            namespaces = {'pom': namespace}
            build = root.find('pom:build', namespaces)
            if build:
                plugins = build.find('pom:plugins', namespaces)
                if plugins:
                    for plugin in plugins:
                        plugin_group_id = plugin.find('pom:groupId', namespaces)
                        plugin_artifact_id = plugin.find('pom:artifactId', namespaces)
                        if (plugin_group_id is not None and plugin_group_id.text == group_id) and \
                           (plugin_artifact_id is not None and plugin_artifact_id.text == artifact_id):
                            return True
            return False
        except ET.ParseError:
            return False
    
    @staticmethod
    def _remove_plugin(pom_file: Path, group_id: str, artifact_id: str):
        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()
            namespace = root.tag.split('}')[0][1:]
            ET.register_namespace('', namespace)
            namespaces = {'pom': namespace}
            build = root.find('pom:build', namespaces)
            if build:
                plugins = build.find('pom:plugins', namespaces)
                if plugins:
                    for plugin in list(plugins):
                        plugin_group_id = plugin.find('pom:groupId', namespaces)
                        plugin_artifact_id = plugin.find('pom:artifactId', namespaces)
                        if (plugin_group_id is not None and plugin_group_id.text == group_id) and \
                           (plugin_artifact_id is not None and plugin_artifact_id.text == artifact_id):
                            plugins.remove(plugin)
            tree.write(pom_file, encoding='utf-8', xml_declaration=True)
        except ET.ParseError as e:
            logger.error(f"Error parsing POM file: {e}")

    @staticmethod
    def _add_plugin(pom_file_path: Path, plugin_def: Dict):
        try:
            tree = ET.parse(pom_file_path)
            root = tree.getroot()
            namespace = root.tag.split('}')[0][1:]
            ET.register_namespace('', namespace)
            namespaces = {'pom': namespace}
            build = root.find('pom:build', namespaces)
            if build is None:
                build = ET.SubElement(root, f'{{{namespace}}}build')
                plugins = ET.SubElement(build, f'{{{namespace}}}plugins')
            else:
                plugins = build.find('pom:plugins', namespaces)
                if plugins is None:
                    plugins = ET.SubElement(build, f'{{{namespace}}}plugins')

            new_plugin = ET.Element(f'{{{namespace}}}plugin')

            groupId = ET.SubElement(new_plugin, f'{{{namespace}}}groupId')
            groupId.text = plugin_def['groupId']

            artifactId = ET.SubElement(new_plugin, f'{{{namespace}}}artifactId')
            artifactId.text = plugin_def['artifactId']

            version = ET.SubElement(new_plugin, f'{{{namespace}}}version')
            version.text = plugin_def['version']

            if 'executions' in plugin_def:
                executions = ET.SubElement(new_plugin, f'{{{namespace}}}executions')
                for exec_data in plugin_def['executions']:
                    execution = ET.SubElement(executions, f'{{{namespace}}}execution')
                    if 'id' in exec_data:
                        id_elem = ET.SubElement(execution, f'{{{namespace}}}id')
                        id_elem.text = exec_data['id']
                    if 'phase' in exec_data:
                        phase = ET.SubElement(execution, f'{{{namespace}}}phase')
                        phase.text = exec_data['phase']
                    goals = ET.SubElement(execution, f'{{{namespace}}}goals')
                    for goal in exec_data['goals']:
                        goal_elem = ET.SubElement(goals, f'{{{namespace}}}goal')
                        goal_elem.text = goal
            else:
                raise ValueError("No executions provided in the plugin definition.")
            
            def _add_configuration(parent, config):
                if isinstance(config, dict):
                    for key, value in config.items():
                        child = ET.SubElement(parent, f'{{{namespace}}}{key}')
                        
                        # if is list, create multiple subelements
                        if isinstance(value, list):
                            for item in value:
                                child.text = str(item)
                        else:
                            _add_configuration(child, value)
                else:
                    parent.text = str(config)
                
            if 'configuration' in plugin_def:
                configuration = ET.SubElement(new_plugin, f'{{{namespace}}}configuration')
                for key, value in plugin_def['configuration'].items():
                    config_elem = ET.SubElement(configuration, f'{{{namespace}}}{key}')
                    _add_configuration(config_elem, value)
            
            
            plugins.append(new_plugin)
            tree.write(pom_file_path, encoding='utf-8', xml_declaration=True)
        except ET.ParseError as e:
            logger.error(f"Error parsing POM file: {e}")
        except ValueError as e:
            logger.error(f"Error adding plugin: {e}")




class GradleBuildSystem(BuildSystem):
    def __init__(self, gradle_file: Path, build_working_dir: Path):
        super().__init__(gradle_file, build_working_dir)
        
    def get_name(self) -> str:
        return "Gradle"

    def execute(self, test_class_identifier: Dict, test_function_identifier: Dict) -> List[str]:
        gradle_executable = "gradlew" if os.path.exists(self.build_working_dir / "gradlew") else "gradle"
        cmd = [gradle_executable, "test", f"--tests={test_class_identifier}#{test_function_identifier}"]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                text=True,               # Decode output to string
                check=True,               # Raise exception on non-zero exit codes
                cwd=self.build_working_dir,  # Set working directory
                timeout=60
            )
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return e.stdout, e.stderr, e.returncode
        
    def clean(self) -> None:
        gradle_executable = "gradlew" if os.path.exists(self.build_working_dir / "gradlew") else "gradle"
        cmd = [gradle_executable, "clean"]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE,  # Capture standard error
                text=True,               # Decode output to string
                check=True,               # Raise exception on non-zero exit codes
                cwd=self.build_working_dir,  # Set working directory
                timeout=60
            )
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return e.stdout, e.stderr, e.returncode

    def get_results(self, test_candidate: TestCandidateDescriptor) -> Optional[Dict]:
        raise NotImplementedError


    def install_tool(self, coverage_tool: 'CoverageTool'):
        raise NotImplementedError
        #self.coverage_tool = coverage_tool
        
    def _add_plugin(self, gradle_file_path: Path, plugin_def: Dict):
        raise NotImplementedError
    
    def _remove_plugin(self, gradle_file: Path, group_id: str, artifact_id: str):
        raise NotImplementedError
    
    def _is_gradle_plugin_installed(self, gradle_file: Path, group_id: str, artifact_id: str) -> bool:
        raise NotImplementedError

class TestExecutor:
    def __init__(self, repo):
        self.repo = repo
        
        self.unit_test = None
        self.test_class_code = None
        
        self.focal_method = None
        self.focal_class_code = None
        
        self.build_system: Optional[BuildSystem] = None
        self.coverage_tool: Optional[CoverageTool] = None
        self.mutation_tool: Optional[MutationTool] = None

    def register_unit_test(self, unit_test: TestCandidateDescriptor):
        self.unit_test = unit_test
        path = Path(self.repo.working_dir) / unit_test.file
        with open(path, "r") as f:
            self.test_class_code = f.read()

    def register_focal_method(self, focal_method: FocalMethodDescriptor):
        self.focal_method = focal_method
        path = Path(self.repo.working_dir) / focal_method.file
        with open(path, "r") as f:
            self.focal_class_code = f.read()

    def detect_build_tool(self) -> Optional[str]:
        build_files = self._find_build_files(self.repo.working_dir)
        path = Path(self.repo.working_dir) / self.unit_test.file
        closest_build_file = self.find_closest_build_file(path, build_files)
        if closest_build_file:
            self.build_system = self._create_build_system(closest_build_file)
            return self.build_system.__class__.__name__
        else:
            raise Exception("No build file found")

    def _create_build_system(self, build_file: Path) -> BuildSystem:
        if build_file.name == "pom.xml":
            return MavenBuildSystem(build_file, build_file.parent)
        elif build_file.name == "build.gradle":
            return GradleBuildSystem(build_file, build_file.parent)
        raise ValueError(f"Unsupported build file: {build_file.name}")


    def find_closest_build_file(self, file_path: Path, build_files: List[Path]) -> Optional[Path]:
        file_path = file_path.resolve()
        build_files = [f.resolve() for f in build_files]
        
        current_dir = file_path.parent
        while current_dir!= current_dir.parent:  # Stop at the filesystem root
            for build_file in build_files:
                if build_file.parent == current_dir:
                    return build_file
            current_dir = current_dir.parent
        return None

    def _find_build_files(self, root_dir: str) -> List[Path]:
        root_path = Path(root_dir)
        return list(root_path.rglob("pom.xml")) + list(root_path.rglob("build.gradle"))

    def execute(self) -> List[str]:
        if not self.build_system: 
            self.detect_build_tool()
        if not self.coverage_tool:
            raise Exception("Coverage tool has not been installed.")
        return self.build_system.execute(self.unit_test.class_identifier, self.unit_test.function_identifier)

    def clean(self) -> None:
        if not self.build_system:
            self.detect_build_tool()
        if self.build_system:
            self.build_system.clean()

    def get_results(self) -> Optional[Dict]:
        if not self.unit_test:
            raise Exception("Unit test has not been registered.")
        if self.build_system:
            return self.build_system.get_results(self.unit_test)
        else:
            raise Exception("Build system has not been detected.")

    def get_coverage_report(self) -> Optional[Dict]:
        if not self.focal_method:
            raise Exception("Focal method has not been registered.")
        if self.coverage_tool:
            return self.coverage_tool.get_report(self.focal_method)
        else:
            logger.warning("Coverage tool has not been installed.")
            return None

    def install_coverage_tool(self, coverage_tool: Optional[CoverageTool] = None):
        if not self.build_system:
            self.detect_build_tool()
        if coverage_tool:
            self.coverage_tool = coverage_tool
        elif isinstance(self.build_system, MavenBuildSystem):
            self.coverage_tool = JacocoMavenCoverageTool(self.build_system.build_working_dir)
        elif isinstance(self.build_system, GradleBuildSystem):
            self.coverage_tool = JacocoGradleCoverageTool(self.build_system.build_working_dir)
        else:
            raise Exception(f"No default coverage tool available for {self.build_system.__class__.__name__}")
        self.build_system.install_tool(self.coverage_tool)
    
    def get_mutation_report(self) -> Optional[Dict]:
        if not self.focal_method:
            raise Exception("Focal method has not been registered.")
        if not self.unit_test:
            raise Exception("Unit test has not been registered.")
        if self.mutation_tool:
            return self.mutation_tool.get_report(self.focal_method, self.unit_test)
        else:
            logger.warning("Mutation tool has not been installed.")
            return None
    
    def install_mutation_tool(self, mutation_tool: Optional[MutationTool] = None):
        if not self.build_system:
            self.detect_build_tool()
        if mutation_tool:
            self.mutation_tool = mutation_tool
        elif isinstance(self.build_system, MavenBuildSystem):
            self.mutation_tool = PITMavenMutationTool(self.build_system.build_working_dir)
        else:
            raise Exception(f"No default mutation tool available for {self.build_system.__class__.__name__}")
        self.build_system.install_tool(self.mutation_tool)

    
    def replace_test_case(self, old_body, new_body) -> None:
        logger.info("Replacing test case")
        
        # open file and read content
        file = self.test_class_code
        
        body_start = file.index(old_body)
        body_end = body_start + len(old_body)
        file2 = file[:body_start] + new_body + file[body_end:]
        
        # overwrite the file
        path = Path(self.repo.working_dir) / self.unit_test.file
        with open(path, "w") as f:
            f.write(file2)
        
    def reset_test_class(self) -> None:
        logger.info("Resetting test class")        
        # overwrite the file
        path = Path(self.repo.working_dir) / self.unit_test.file
        with open(path, "w") as f:
            f.write(self.test_class_code)
