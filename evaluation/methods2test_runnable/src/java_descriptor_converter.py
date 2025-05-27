import re

class JavaDescriptorConverter:
    def __init__(self):
        self.primitive_map = {
            'Z': 'boolean', 'B': 'byte', 'C': 'char', 'D': 'double',
            'F': 'float', 'I': 'int', 'J': 'long', 'S': 'short', 'V': 'void'
        }
        self.primitive_map_reverse = {
            'boolean': 'Z', 'byte': 'B', 'char': 'C', 'double': 'D',
            'float': 'F', 'int': 'I', 'long': 'J', 'short': 'S', 'void': 'V'
        }

    def _parse_type(self, descriptor: str) -> tuple[str, str]:
        """Parses a single type descriptor and returns the Java type and the remaining descriptor."""
        if descriptor.startswith('L'):
            match = re.match(r'L([^;]+);(.*)', descriptor)
            if match:
                class_name = f'L{match.group(1).split("/")[-1]};'  # Keep the 'L' and ';'
                remaining = match.group(2)
                return class_name, remaining
        elif descriptor.startswith('['):
            component_type, remaining = self._parse_type(descriptor[1:])
            return f'[{component_type}', remaining
        elif descriptor[0] in self.primitive_map:
            return descriptor[0], descriptor[1:]
        elif descriptor.startswith('T'):
            match = re.match(r'T([^;]+);(.*)', descriptor)
            if match:
                type_variable = 'TTypeVariable;' # Keep 'T' and ';' for type variables
                remaining = match.group(2)
                return type_variable, remaining
        elif descriptor.startswith('+') or descriptor.startswith('-') or descriptor.startswith('*'):
            # Handle wildcard type arguments
            bound_type, remaining = self._parse_type(descriptor[1:]) if len(descriptor) > 1 else ('*', '')
            return '*', remaining
        return descriptor, ""

    def descriptor_to_signature(self, descriptor: str) -> str:
        """Converts a Java method descriptor to a human-readable signature without parameter names."""
        match = re.match(r'\((.*)\)(.*)', descriptor)
        if not match:
            raise ValueError(f"Invalid method descriptor: {descriptor}")

        arguments_descriptor = match.group(1)
        return_type_descriptor = match.group(2)

        arguments = []
        remaining_args = arguments_descriptor
        while remaining_args:
            arg_type, remaining_args = self._parse_type(remaining_args)
            java_type = self.primitive_map.get(arg_type, arg_type.replace('L', '').replace(';', '').replace('/', '.'))
            arguments.append(java_type)

        return_type_raw, _ = self._parse_type(return_type_descriptor)
        return_type = self.primitive_map.get(return_type_raw, return_type_raw.replace('L', '').replace(';', '').replace('/', '.'))

        # Extract method name (best effort - not part of the descriptor)
        method_name_match = re.match(r'([a-zA-Z_$][a-zA-Z0-9_$]*)?\((.*)\).*', descriptor)
        method_name = method_name_match.group(1) if method_name_match and method_name_match.group(1) else "method"

        return f'{return_type} {method_name}({", ".join(arguments)})'

    def class_descriptor_to_signature(self, descriptor: str) -> str:
        """Converts a Java class descriptor to a human-readable signature with generics."""
        if descriptor.startswith('L') and descriptor.endswith(';'):
            class_path = descriptor[1:-1].replace('/', '.')
            # Attempt to handle simple generic class signatures (e.g., List<String>)
            if "<" in class_path:
                return class_path
            else:
                return class_path
        elif descriptor.startswith('['):
            component_type = self.class_descriptor_to_signature(descriptor[1:])
            return f'{component_type}[]'
        elif descriptor[0] in self.primitive_map:
            return self.primitive_map[descriptor[0]]
        elif descriptor.startswith('T') and descriptor.endswith(';'):
            return descriptor[1:-1] # Type variable
        return descriptor


    def _type_to_descriptor(self, type_signature: str) -> str:
        """Converts a Java type signature part to its descriptor (without generic details)."""
        if type_signature in self.primitive_map_reverse:
            return self.primitive_map_reverse[type_signature]
        elif type_signature.endswith('[]'):
            component_type = type_signature[:-2]
            return f'[{self._type_to_descriptor(component_type)}'
        else:
            # For object types, only the base class is in the descriptor
            base_type = type_signature.split('<')[0] if '<' in type_signature else type_signature
            return f'L{base_type.replace(".", "/")};'

    def signature_to_descriptor(self, signature: str) -> str:
        """Converts a Java method signature to its descriptor, ignoring parameter names."""
        match = re.match(r'(?:<.+?>\s+)?([^\s]+)\s+[^\(]+\((.*)\)', signature)
        if not match:
            raise ValueError(f"Invalid method signature: {signature}")

        return_type_str = match.group(1)
        arguments_str = match.group(2)

        argument_descriptors = []
        if arguments_str:
            args = [arg.strip() for arg in re.split(r',\s*(?!(?:[^<>]|<[^<>]*>)*\))', arguments_str)]
            for arg in args:
                parts = arg.split()
                # Extract type signature, ignoring modifiers and name (last part)
                type_sig_parts = [p for p in parts if p not in ['final', 'volatile', 'transient', 'static', 'native', 'synchronized']]
                if type_sig_parts:
                    type_sig = ' '.join(type_sig_parts[:-1] if len(type_sig_parts) > 1 else type_sig_parts)
                    argument_descriptors.append(self._type_to_descriptor(type_sig.strip()))

        return_type_descriptor = self._type_to_descriptor(return_type_str)
        return f'({"".join(argument_descriptors)}){return_type_descriptor}'

    def get_generic_signature_for_type(self, type_signature: str) -> str:
        """Returns the generic type signature for a given Java type signature."""
        if type_signature.startswith('java.util.List<') and type_signature.endswith('>'):
            inner_type = type_signature[len('java.util.List<'):-1]
            if inner_type.endswith('[]'):
                component_type = inner_type[:-2]
                if component_type.startswith('java.util.List<') and component_type.endswith('>'):
                    grand_inner_type = component_type[len('java.util.List<'):-1]
                    return f'Ljava/util/List<[Ljava/util/List<L{grand_inner_type.replace(".", "/")};>;>;'
        return f'L{type_signature.replace(".", "/")};' # Default if no specific generic case handled

    def _simplify_type_signature(self, type_signature: str) -> str:
        """Simplifies a Java type signature for shallow comparison, removing modifiers."""
        parts = type_signature.split()
        type_without_modifiers = ' '.join(p for p in parts if p not in ['final', 'volatile', 'transient', 'static', 'native', 'synchronized'])

        if '<' in type_without_modifiers:
            base_type = type_without_modifiers.split('<')[0]
            if base_type in self.primitive_map_reverse:
                return base_type
            else:
                return base_type
        elif type_without_modifiers.endswith('[]'):
            base_type = type_without_modifiers[:-2]
            if base_type in self.primitive_map_reverse:
                return f'{base_type}[]'
            else:
                return 'Object[]'
        elif type_without_modifiers in self.primitive_map_reverse:
            return type_without_modifiers
        else:
            return type_without_modifiers

    def normalize(self, signature: str) -> str:
        """
        Normalizes a Java method signature or descriptor to a shallow representation,
        removing type modifiers and parameter names from signatures and preserving descriptor format for descriptors.
        """
        if '(' in signature and ')' in signature:  # Likely a method signature or descriptor
            if signature.startswith('('):  # It's a descriptor
                match = re.match(r'\((.*)\)(.*)', signature)
                if not match:
                    raise ValueError(f"Invalid method descriptor: {signature}")

                arguments_descriptor = match.group(1)
                return_type_descriptor = match.group(2)

                arguments = []
                remaining_args = arguments_descriptor
                while remaining_args:
                    arg_type, remaining_args = self._parse_type(remaining_args)
                    arguments.append(arg_type)

                return_type, _ = self._parse_type(return_type_descriptor)
                return f'({",".join(arguments)}){return_type}'
            else:  # It's a method signature
                match = re.match(r'(?:<.+?>\s+)?([^\s]+)\s+[^\(]+\((.*)\)', signature)
                if not match:
                    raise ValueError(f"Invalid method signature: {signature}")

                return_type_str = match.group(1)
                arguments_str = match.group(2)

                argument_types = []
                if arguments_str:
                    args = [arg.strip() for arg in re.split(r',\s*(?!(?:[^<>]|<[^<>]*>)*\))', arguments_str)]
                    for arg in args:
                        parts = arg.split()
                        # Remove type modifiers and the last part (parameter name)
                        type_sig_parts = [p for p in parts if p not in ['final', 'volatile', 'transient', 'static', 'native', 'synchronized']]
                        if type_sig_parts:
                            type_sig = ' '.join(type_sig_parts[:-1] if len(type_sig_parts) > 1 else type_sig_parts)
                            argument_types.append(self._simplify_type_signature(type_sig.strip()))

                return_type_shallow = self._simplify_type_signature(return_type_str)
                return f'({",".join(argument_types)}){return_type_shallow}'
        else:  # It's a type descriptor
            return self._parse_type(signature)[0]

    def are_shallowly_equal(self, sig1: str, sig2: str) -> bool:
        """Checks if two method signatures or descriptors are shallowly equal."""
        return self.normalize(sig1) == self.normalize(sig2)