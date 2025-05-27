import xml.etree.ElementTree as ET
import pandas as pd
from .java_descriptor_converter import JavaDescriptorConverter

class JaCoCoReport:
    def __init__(self, xml_string):
        self.root = ET.fromstring(xml_string)

    def get_session_info(self):
        session_info_list = []
        for sessioninfo in self.root.findall('sessioninfo'):
            session_info = {
                'id': sessioninfo.attrib['id'],
              'start': sessioninfo.attrib['start'],
                'dump': sessioninfo.attrib['dump']
            }
            session_info_list.append(session_info)
        return session_info_list

    def get_packages(self):
        package_list = []
        for package in self.root.findall('package'):
            package_info = {
                'name': package.attrib['name'],
                'classes': self.get_classes(package)
            }
            package_list.append(package_info)
        return package_list

    def get_classes(self, package):
        class_list = []
        for cls in package.findall('class'):
            class_info = {
                'name': cls.attrib['name'],
              'sourcefilename': cls.attrib.get('sourcefilename'),
              'methods': self.get_methods(cls)
            }
            class_list.append(class_info)
        return class_list

    def get_methods(self, cls):
        method_list = []
        for method in cls.findall('method'):
            method_info = {
                'name': method.attrib['name'],
                'desc': method.attrib['desc'],
                'line': method.attrib.get('line'),
                'counters': self.get_counters(method)
            }
            method_list.append(method_info)
        return method_list

    def get_counters(self, element):
        counter_list = []
        for counter in element.findall('counter'):
            counter_info = {
                'type': counter.attrib['type'],
              'missed': int(counter.attrib['missed']),
                'covered': int(counter.attrib['covered'])
            }
            counter_list.append(counter_info)
        return counter_list

    def get_group(self):
        group_list = []
        for group in self.root.findall('group'):
            group_info = {
                'name': group.attrib['name'],
                'groups': self.get_group(group),
                'packages': self.get_packages(group)
            }
            group_list.append(group_info)
        return group_list

    def to_dataframe(self):
        data = []
        for package in self.get_packages():
            for cls in package['classes']:
                for method in cls['methods']:
                    row = {
                        'GROUP': '',  # Not available in the provided DTD
                        'PACKAGE': package['name'],
                        'CLASS': cls['name'],
                        'METHOD_NAME': method['name'],
                        'METHOD_DESC': method['desc'],
                        'INSTRUCTION_MISSED': 0,
                        'INSTRUCTION_COVERED': 0,
                        'BRANCH_MISSED': 0,
                        'BRANCH_COVERED': 0,
                        'LINE_MISSED': 0,
                        'LINE_COVERED': 0,
                        'COMPLEXITY_MISSED': 0,
                        'COMPLEXITY_COVERED': 0,
                        'METHOD_MISSED': 0,
                        'METHOD_COVERED': 0,
                    }
                    for counter in method['counters']:
                        if counter['type'] == 'INSTRUCTION':
                            row['INSTRUCTION_MISSED'] = counter['missed']
                            row['INSTRUCTION_COVERED'] = counter['covered']
                        elif counter['type'] == 'BRANCH':
                            row['BRANCH_MISSED'] = counter['missed']
                            row['BRANCH_COVERED'] = counter['covered']
                        elif counter['type'] == 'LINE':
                            row['LINE_MISSED'] = counter['missed']
                            row['LINE_COVERED'] = counter['covered']
                        elif counter['type'] == 'COMPLEXITY':
                            row['COMPLEXITY_MISSED'] = counter['missed']
                            row['COMPLEXITY_COVERED'] = counter['covered']
                        elif counter['type'] == 'METHOD':
                            row['METHOD_MISSED'] = counter['missed']
                            row['METHOD_COVERED'] = counter['covered']
                    data.append(row)
        return pd.DataFrame(data)
    
    def match_method(self, package_name, class_name, method_name, normalized_descriptor):
        converter = JavaDescriptorConverter()
        
        class_path = (package_name + '.' + class_name).replace('.', '/')
        package_name = package_name.replace('.', '/')
        
        
        for package in self.get_packages():
            if package['name'] == package_name:
                for cls in package['classes']:
                    if cls['name'] == class_path:
                        for method in cls['methods']:
                            if method['name'] == method_name:
                                if converter.normalize(method['desc']) == normalized_descriptor:
                                    data = {
                                        'PACKAGE': package['name'],
                                        'CLASS': cls['name'],
                                        'METHOD_NAME': method['name'],
                                        'METHOD_DESC': method['desc'],
                                        'INSTRUCTION_MISSED': 0,
                                        'INSTRUCTION_COVERED': 0,
                                        'BRANCH_MISSED': 0,
                                        'BRANCH_COVERED': 0,
                                        'LINE_MISSED': 0,
                                        'LINE_COVERED': 0,
                                        'COMPLEXITY_MISSED': 0,
                                        'COMPLEXITY_COVERED': 0,
                                        'METHOD_MISSED': 0,
                                        'METHOD_COVERED': 0,
                                    }
                                    for counter in method['counters']:
                                        if counter['type'] == 'INSTRUCTION':
                                            data['INSTRUCTION_MISSED'] = counter['missed']
                                            data['INSTRUCTION_COVERED'] = counter['covered']
                                        elif counter['type'] == 'BRANCH':
                                            data['BRANCH_MISSED'] = counter['missed']
                                            data['BRANCH_COVERED'] = counter['covered']
                                        elif counter['type'] == 'LINE':
                                            data['LINE_MISSED'] = counter['missed']
                                            data['LINE_COVERED'] = counter['covered']
                                        elif counter['type'] == 'COMPLEXITY':
                                            data['COMPLEXITY_MISSED'] = counter['missed']
                                            data['COMPLEXITY_COVERED'] = counter['covered']
                                        elif counter['type'] == 'METHOD':
                                            data['METHOD_MISSED'] = counter['missed']
                                            data['METHOD_COVERED'] = counter['covered']
                                    
                                    return data
        return None
    