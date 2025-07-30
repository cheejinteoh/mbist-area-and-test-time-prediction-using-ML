import json
import re
import os
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configuration
DATASET_PATH = "verilog_signals.json"
MODEL_DIR = "trained_model"
RANDOM_STATE = 42
BACKUP_DIR = "training_history"
PROJECT_ROOT = os.getcwd()

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)


class DesignAnalyzer:
    def __init__(self):
        self.classifier = PortClassifier()
        self.modules = {}  # Stores module info
        self.addr_bit_width = {}   # {module_name: {port: width, ...}}
        self.data_in_bit_width = {}
        self.data_out_bit_width = {}
        self.processed_modules = set()   # To avoid duplicate processing
        self.warned_modules = set()      # To warn missing submodules only once

    def analyze_design(self, design_file):
        """Run the analysis on the given design file."""
        self._process_file(design_file, is_top=True)

    def _process_file(self, file_path, parent_module=None, is_top=False):
        """Recursively process Verilog/SystemVerilog files."""
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        module_info = self._parse_verilog_file(file_path)
        module_name = module_info["module_name"]
        if module_name in self.processed_modules:
            return
        self.processed_modules.add(module_name)

        module_type = "Top Module" if is_top else "Sub-Module"
        print(f"Found {module_type}: {module_name}")

        # Classify ports using the classifier model.
        port_results = []
        for port_info in module_info["ports"]:
            pred, confidence = self.classifier.predict_port(port_info["name"])
            port_results.append((port_info["name"], pred, confidence))
            lower_name = port_info["name"].lower()
            # For bitwidth info, we use our _get_bit_width method.
            if "addr" in lower_name or "address" in lower_name or lower_name == "a":
                self.addr_bit_width.setdefault(module_name, {})[port_info["name"]] = self._get_bit_width(file_path, port_info, module_info['parameters'])
            elif "data_in" in lower_name or lower_name == "d":
                self.data_in_bit_width.setdefault(module_name, {})[port_info["name"]] = self._get_bit_width(file_path, port_info, module_info['parameters'])
            elif "data_out" in lower_name or lower_name == "q":
                self.data_out_bit_width.setdefault(module_name, {})[port_info["name"]] = self._get_bit_width(file_path, port_info, module_info['parameters'])
            elif "clk" in lower_name:
                # Clock width is always 1; dimension info is handled in bitwidth computation.
                pass

        self.modules[module_name] = {
            "type": module_type,
            "ports": port_results,
            "bitwidths": module_info.get("bitwidths", {}),
            "instances": []
        }

        # Process submodules.
        for instance in module_info["instances"]:
            submodule_name = instance["module_type"]
            submodule_file = self._find_submodule_file(submodule_name)
            if submodule_file:
                self._process_file(submodule_file, parent_module=module_name)
                if submodule_name not in self.modules[module_name]["instances"]:
                    self.modules[module_name]["instances"].append(submodule_name)

    def _parse_verilog_file(self, file_path):
        """
        Parse a Verilog/SystemVerilog file and extract:
          - Module name
          - List of ports (each as a dict with keys "name" and "dims")
          - List of instantiated submodules
          - Module parameters (for bit‑width calculations)
          - Compute bit‑width info for each port (for clock, address, data_in, data_out)
        """
        with open(file_path, "r") as f:
            content = f.read()

        # Remove multiline comments.
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        parameters = self._extract_parameters(content)
        module_match = re.search(r"module\s+(\w+)\s*(?:#\s*\((.*?)\))?\s*\((.*?)\)\s*;", content, re.DOTALL)
        module_name = module_match.group(1) if module_match else "unknown"
        port_list = module_match.group(3) if module_match else ""
        port_list = re.sub(r"//.*", "", port_list)
        ports = self._extract_ports(port_list)
        ports = self._resolve_parameters_in_ports(ports, parameters)

        # Compute bitwidth info.
        calc_bitwidths = {}
        for port in ports:
            pname = port["name"]
            dims = port["dims"]
            if "clk" in pname.lower():
                # Clock handling - width is always 1
                total_dim = 1
                for d in dims:
                    total_dim *= self._calc_bracket_width(d)
                calc_bitwidths[pname] = {"dim": total_dim, "width": 1}
            elif dims:
                # For non-clock ports with multiple dimensions
                element_width = self._calc_bracket_width(dims[-1])
                total_dim = 1
                for d in dims[:-1]:
                    total_dim *= self._calc_bracket_width(d)
                calc_bitwidths[pname] = {"dim": total_dim, "width": element_width}
            else:
                # Scalar port
                if pname == 'A':
                    width = parameters.get("ROW_BITS", 5) + parameters.get("COL_BITS", 2)
                    calc_bitwidths[pname] = {"dim": 1, "width": width}
                elif pname in ['D', 'Q']:
                    width = parameters.get("IO", 16)
                    calc_bitwidths[pname] = {"dim": 1, "width": width}
                else:
                    calc_bitwidths[pname] = {"dim": 1, "width": 1}

        # Extract instantiations.
        instances = []
        instantiation_pattern = re.compile(r"(?sm)^\s*(\w+)\s+(\w+)\s*\(.*?\);")
        for inst in instantiation_pattern.findall(content):
            if inst[0] == module_name:
                continue
            instances.append({
                "module_type": inst[0],
                "instance_name": inst[1]
            })

        return {
            "module_name": module_name,
            "ports": ports,
            "instances": instances,
            "parameters": parameters,
            "bitwidths": calc_bitwidths
        }

    def _calc_bracket_width(self, bracket_content):
        """
        Given a bracket content string like "31:0", compute the width as (upper - lower + 1).
        Handles expressions and also single numbers.
        """
        try:
            # Split by colon and strip each part
            parts = [part.strip() for part in bracket_content.split(":")]
            if len(parts) == 2:
                # Evaluate each part
                upper_val = eval(parts[0])
                lower_val = eval(parts[1])
                return abs(upper_val - lower_val) + 1
            else:
                # It's a single number or expression
                value = eval(bracket_content.strip())
                return abs(value)
        except Exception as e:
            print(f"Warning: Could not compute bracket width for '{bracket_content}': {e}. Using default 16.")
            return 16

    def _extract_parameters(self, content):
        """
        Extract parameters from Verilog code and evaluate them.
        This regex stops at a comma or closing parenthesis.
        """
        parameters = {}
        param_match = re.findall(r"parameter\s+(\w+)\s*=\s*([^;\)]+)", content)
        for param_name, param_value in param_match:
            try:
                clean_value = param_value.strip()
                parameters[param_name.strip()] = eval(clean_value)
            except Exception as e:
                print(f"Warning: Could not evaluate parameter {param_name} with value {param_value}. Error: {e}")
        return parameters

    def _resolve_parameters_in_ports(self, ports, parameters):
        """
        For each port (a dict with "name" and "dims"), substitute parameter names with their evaluated values
        and remove common keywords like 'input', 'output', and 'logic'. Also update dimension strings.
        """
        resolved = []
        for port in ports:
            name = port["name"]
            name = re.sub(r"^(input|output|inout)\s+logic\s*", "", name, flags=re.IGNORECASE)
            name = re.sub(r"^(input|output|inout)\s+", "", name, flags=re.IGNORECASE)
            for param_name, param_value in parameters.items():
                name = re.sub(rf"\b{param_name}\b", str(param_value), name)
            port["name"] = name.strip()
            new_dims = []
            for d in port["dims"]:
                for param_name, param_value in parameters.items():
                    d = re.sub(rf"\b{param_name}\b", str(param_value), d)
                new_dims.append(d.strip())
            port["dims"] = new_dims
            resolved.append(port)
        return resolved

    def _clean_line(self, line):
        """Remove single-line comments and trim whitespace."""
        line = re.sub(r"//.*", "", line)
        line = line.strip()
        return "" if not line or line.startswith("//") else line

    def _extract_ports(self, port_list):
        """
        Extract port declarations from the module's port list.
        Returns a list of dicts with:
          - "name": the cleaned port name
          - "dims": a list of dimension strings (contents inside square brackets)
        """
        ports = []
        for port in re.split(r"\s*,\s*", port_list):
            port = port.strip()
            if not port:
                continue
            port = re.sub(r"//.*", "", port)
            dims = re.findall(r"\[([^\]]+)\]", port)
            name = re.sub(r"\s*\[.*?\]\s*", "", port)
            name = re.sub(r"\s*=\s*.*", "", name)
            ports.append({"name": name.strip(), "dims": dims})
        return ports

    def _find_submodule_file(self, module_name):
        """
        Search for a submodule file in the project directory.
        If not found, print a warning once for that module name.
        """
        for ext in [".v", ".sv"]:
            possible_path = os.path.join(PROJECT_ROOT, f"{module_name}{ext}")
            if os.path.exists(possible_path):
                return possible_path
        if module_name not in self.warned_modules:
            print(f"Warning: Could not find file for module {module_name}")
            self.warned_modules.add(module_name)
        return None

    def _get_bit_width(self, file_path, port_info, parameters):
        """
        Determine the bit-width for a given port using module parameters.
        For ports 'D' and 'Q', if dimensions exist, use the last bracket group.
        For port 'A', compute ROW_BITS + COL_BITS.
        For clock ports, return 1 (the dimension info is in bitwidths).
        """
        if "clk" in port_info["name"].lower():
            return 1
        with open(file_path, "r") as f:
            content = f.read()
        dims = port_info["dims"]
        if port_info["name"] in ['D', 'Q']:
            if dims:
                return self._calc_bracket_width(dims[-1])
            else:
                return parameters.get("IO", 16)
        elif port_info["name"] == 'A':
            return parameters.get("ROW_BITS", 5) + parameters.get("COL_BITS", 2)
        else:
            return 1

    def _print_results(self):
        """Print the design analysis results in original formatting."""
        print("\n\n=== Design Analysis Results ===")
        for module_name, data in self.modules.items():
            print(f"\nModule: {module_name} ({data['type']})")
            for port, pred, conf in data["ports"]:
                print(f"  Port: {port} -> {pred} (Confidence: {conf:.2f})")
            if data["instances"]:
                print("  Instantiates:", ", ".join(data["instances"]))


class PortClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
        self.model = LogisticRegression(class_weight="balanced",
                                        random_state=RANDOM_STATE,
                                        max_iter=1000)
        self.labels = ["clock", "address", "data_in", "data_out", "other"]
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """Load and normalize dataset from JSON."""
        try:
            with open(DATASET_PATH, "r") as f:
                data = json.load(f)
            for item in data:
                item["code"] = item["code"].lower()
            return data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def save_dataset(self):
        """Save the dataset with a backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"backup_{timestamp}.json")
        with open(backup_path, "w") as f:
            json.dump(self.dataset, f, indent=2)
        with open(DATASET_PATH, "w") as f:
            json.dump(self.dataset, f, indent=2)

    def train(self):
        """Train the model using the dataset."""
        if not self.dataset:
            print("No training data available!")
            return
        texts = [item["code"] for item in self.dataset]
        labels = [item["label"] for item in self.dataset]
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.save_model()
        X_train, X_test, y_train, y_test = train_test_split(X, labels,
                                                            test_size=0.2,
                                                            random_state=RANDOM_STATE)
        y_pred = self.model.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))

    def save_model(self):
        """Save the trained model and vectorizer."""
        with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
                self.model = pickle.load(f)
            with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            print("Loaded pre-trained model.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict_port(self, port_name):
        """Predict the port category for a given port name."""
        port_lower = port_name.lower()
        vec = self.vectorizer.transform([port_lower])
        proba = self.model.predict_proba(vec)[0]
        prediction = self.model.predict(vec)[0]
        return prediction, max(proba)

    def add_correction(self, port_name, correct_label):
        """Add a correction for a misclassified port."""
        port_lower = port_name.lower()
        self.dataset = [item for item in self.dataset if item["code"] != port_lower]
        self.dataset.append({"code": port_lower, "label": correct_label})
        self.save_dataset()
        print(f"Debug: Added {port_lower} -> {correct_label} to dataset")
        print(f"Debug: Current dataset size: {len(self.dataset)}")


def run_analysis(design_file, interactive=False):
    """Helper function to run analysis on a given design file."""
    analyzer = DesignAnalyzer()
    analyzer.classifier.load_model()
    analyzer.analyze_design(design_file)
    if interactive:
        for module_name, data in analyzer.modules.items():
            for port, pred, conf in data["ports"]:
                if conf < 0.7:
                    print(f"\nLow confidence prediction for {port} in module {module_name}")
                    print("Please select correct label: ")
                    for i, label in enumerate(analyzer.classifier.labels):
                        print(f"{i+1}. {label}")
                    print("5. other (default)")
                    choice = input("Enter choice (1-5): ").strip()
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < 4:
                            correct_label = analyzer.classifier.labels[choice_idx]
                        else:
                            correct_label = "other"
                        analyzer.classifier.add_correction(port, correct_label)
                        print(f"Updated dataset with {port} -> {correct_label}")
                        print("Retraining model...")
                        analyzer.classifier.train()
                        new_pred, new_conf = analyzer.classifier.predict_port(port)
                        print(f"New prediction for {port}: {new_pred} (Confidence: {new_conf:.2f})")
                    except ValueError:
                        print("Invalid input, skipping correction")
    return analyzer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verilog Port Analyzer")
    parser.add_argument("--design_file", help="The design file (e.g., engine.v)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive feedback mode")
    args = parser.parse_args()
    run_analysis(args.design_file, interactive=args.interactive)