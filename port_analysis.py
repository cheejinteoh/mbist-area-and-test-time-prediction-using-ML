import re
import os
from train_and_classify import run_analysis

def read_design_file_from_config(config_file):
    """
    Reads the configuration file and extracts the design file name specified by the 'VerilogFile' field.
    The search is case-insensitive and ignores spacing variations.
    Example: "VerilogFile       : engine.v;" or "verilogfile: engine.v"
    """
    try:
        with open(config_file, "r") as f:
            config_text = f.read()
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return None

    match = re.search(r"(?i)verilogfile\s*:\s*([\w\.]+)", config_text)
    if match:
        return match.group(1).strip()
    else:
        print("VerilogFile not found in config.")
        return None

def remove_comments(text):
    """
    Removes multiline (/* ... */) and single-line (// ...) comments from the text.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text

def extract_block(text, start_keyword):
    """
    Extracts a block of text starting with the given start_keyword and its opening brace '{'.
    Uses a brace-counting mechanism to return the content until the matching closing brace.
    """
    pattern = r"(?i)" + re.escape(start_keyword) + r"\s*\{"
    match = re.search(pattern, text)
    if not match:
        return None
    start = match.end()
    brace_count = 1
    pos = start
    while pos < len(text) and brace_count > 0:
        if text[pos] == "{":
            brace_count += 1
        elif text[pos] == "}":
            brace_count -= 1
        pos += 1
    return text[start:pos-1]

def read_clock_config(config_file):
    """
    Reads the configuration file and extracts active clock names from the ClocksInfo block.
    Removes all comments, extracts the ClocksInfo block using a brace-counting method,
    then finds each Clock { ... } block and extracts the clock name.
    Returns a list of clock names.
    """
    try:
        with open(config_file, "r") as f:
            config_text = f.read()
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return []

    config_text = remove_comments(config_text)
    clocks_info = extract_block(config_text, "ClocksInfo")
    if not clocks_info:
        print("No ClocksInfo block found in config.")
        return []
    
    # Find all Clock { ... } blocks within the ClocksInfo block.
    clock_blocks = re.findall(r"(?is)Clock\s*\{(.*?)\}", clocks_info)
    clock_names = []
    for block in clock_blocks:
        match = re.search(r"(?i)name\s*:\s*([\w\.]+)", block)
        if match:
            clock_names.append(match.group(1).strip())
    return clock_names

def read_memory_config(config_file):
    """
    Reads the configuration file and extracts memory module names from the MemoryInfo block.
    Returns a set of memory module names.
    """
    try:
        with open(config_file, "r") as f:
            config_text = f.read()
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return set()

    config_text = remove_comments(config_text)
    meminfo_match = re.search(r"(?i)MemoryInfo\s*\{(.*?)\}", config_text, re.DOTALL)
    if not meminfo_match:
        return set()
    meminfo = meminfo_match.group(1)
    modules = re.findall(r"(?i)Module\s*:\s*([^;]+);", meminfo)
    mem_modules = set()
    for m in modules:
        for mod in m.split(","):
            mod = mod.strip()
            if mod:
                mem_modules.add(mod)
    return mem_modules

def extract_memory_instantiations(design_file):
    """
    Reads the top-level design file and extracts instantiated memory module names.
    Uses a regex with negative lookahead to ignore lines starting with "module".
    Returns a set of unique memory module names.
    """
    try:
        with open(design_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {design_file}: {e}")
        return set()

    content = remove_comments(content)
    instantiations = re.findall(r"(?m)^(?!\s*module\b)\s*([A-Za-z0-9_]+)\s+\w+\s*\(", content)
    return set(instantiations)

def read_memlib_config(config_file):
    """
    Reads the configuration file and extracts the memory library directory (MemLibDir)
    and memory library extension(s) (MemoryLibraryExtension). The extension field may contain
    multiple extensions separated by commas.
    Returns a tuple: (mem_lib_dir, list_of_extensions).
    """
    try:
        with open(config_file, "r") as f:
            config_text = f.read()
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return (".", [])
    
    config_text = remove_comments(config_text)
    memlib_dir_match = re.search(r"(?i)MemLibDir\s*:\s*([^;]+);", config_text)
    ext_match = re.search(r"(?i)MemoryLibraryExtension\s*:\s*([^;]+);", config_text)
    
    mem_lib_dir = memlib_dir_match.group(1).strip() if memlib_dir_match else "."
    if ext_match:
        ext_text = ext_match.group(1).strip()
        extensions = [ext.strip() for ext in ext_text.split(",")]
    else:
        extensions = []
    
    return (mem_lib_dir, extensions)

def check_memory_library_files(config_file, design_file):
    """
    Checks whether each memory module instantiated in the design file has a corresponding
    memory library file in the directory specified by MemLibDir in block.config.
    The expected file name for a memory module is: <module_name>.<extension>
    where extension is one of the extensions specified in MemoryLibraryExtension.
    Prints a warning for each memory module that does not have a corresponding file.
    """
    mem_lib_dir, extensions = read_memlib_config(config_file)
    if not extensions:
        print("No MemoryLibraryExtension defined in config.")
        return
    mem_instantiated = extract_memory_instantiations(design_file)
    for mem_module in mem_instantiated:
        file_found = False
        for ext in extensions:
            mem_file = os.path.join(mem_lib_dir, f"{mem_module}.{ext}")
            if os.path.exists(mem_file):
                file_found = True
                break
        if not file_found:
            print(f"Warning: Memory module '{mem_module}' does not have a defined memory library file and will not have MBIST logic.")

def extract_instantiation_details(design_file, submodule_name):
    """
    Extracts instantiation details for a given submodule from the top-level design file.
    Returns a list of dictionaries with:
      'instance_name': instance name,
      'connections': dictionary mapping port names to connection strings.
    Only the first instantiation is considered.
    """
    try:
        with open(design_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {design_file}: {e}")
        return []
    content = remove_comments(content)
    pattern = rf"(?i){submodule_name}\s+(\w+)\s*\((.*?)\);"
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return []
    instance_name, connections_block = matches[0]
    connections = dict(re.findall(r"\.(\w+)\s*\(\s*([^)]+)\)", connections_block))
    return [{"instance_name": instance_name.strip(), "connections": connections}]

def get_top_module_bitwidths(analyzer):
    """
    Returns the bitwidth dictionary from the top module.
    """
    for module_name, info in analyzer.modules.items():
        if info["type"].lower() == "top module":
            return info.get("bitwidths", {})
    return {}

def get_connection_width(conn, top_bitwidths):
    """
    Given a connection string (e.g. "16'h0000" or "clk2"), returns its bitwidth.
    If the connection is a constant like 16'h0000, returns the number.
    Otherwise, if it's a signal name, looks it up in top_bitwidths.
    Returns an integer or None if not determinable.
    """
    conn = conn.strip()
    const_match = re.match(r"(\d+)\s*'\s*[hd]\w+", conn, re.IGNORECASE)
    if const_match:
        return int(const_match.group(1))
    return top_bitwidths.get(conn, {}).get("width", None)

def compare_submodule_bitwidths(design_file, analyzer, config_file):
    """
    For each memory submodule (non-top module) that has bitwidth information,
    extract instantiation details from the top-level design file and compare the bitwidths
    for key ports (CLK, A, D, Q) between the submodule's definition, its top-level connection,
    and, if available, the memory library file.
    Only the first instantiation is used for comparison.
    """
    # Get top module bitwidth info.
    top_bitwidths = {}
    for module_name, info in analyzer.modules.items():
        if info["type"].lower() == "top module":
            top_bitwidths = info.get("bitwidths", {})
            break

    # For each submodule (non-top) with bitwidth info.
    for submodule, info in analyzer.modules.items():
        if info["type"].lower() == "top module":
            continue
        if not info.get("bitwidths"):
            continue

        # Attempt to load memory library info for this submodule.
        mem_lib_dir, extensions = read_memlib_config(config_file)
        memlib_ports = {}
        mem_file_found = False
        for ext in extensions:
            candidate = os.path.join(mem_lib_dir, f"{submodule}.{ext}")
            if os.path.exists(candidate):
                mem_file_found = True
                with open(candidate, "r") as f:
                    content = f.read()
                port_pattern = re.compile(r"(?i)port\s*\(\s*([A-Za-z0-9_]+)(\[[^)\]]+\])?\s*\)")
                for signal, bit_range in port_pattern.findall(content):
                    if bit_range:
                        dims = bit_range.strip()[1:-1]  # remove brackets
                        parts = dims.split(":")
                        if len(parts) == 2:
                            try:
                                upper = int(parts[0].strip())
                                lower = int(parts[1].strip())
                                width = abs(upper - lower) + 1
                            except Exception:
                                width = None
                        else:
                            try:
                                width = int(parts[0].strip())
                            except Exception:
                                width = None
                    else:
                        width = 1
                    memlib_ports[signal] = width
                break  # use the first found memory file

        inst_details = extract_instantiation_details(design_file, submodule)
        if not inst_details:
            continue
        print(f"\nComparing bitwidths for submodule '{submodule}':")
        for inst in inst_details:
            instance_name = inst["instance_name"]
            connections = inst["connections"]
            print(f"  Instance: {instance_name}")
            # Compare key ports: CLK, A, D, Q
            key_ports = ["CLK", "A", "D", "Q"]
            for port in key_ports:
                sub_width = info.get("bitwidths", {}).get(port, {}).get("width", None)
                conn = connections.get(port, None)
                conn_width = get_connection_width(conn, top_bitwidths) if conn is not None else None
                mem_width = memlib_ports.get(port, None) if mem_file_found else None
                if sub_width is not None:
                    if mem_file_found:
                        print(f"    Port {port}: OK (Submodule width = {sub_width}, Top Level Connection width = {conn_width}, Memory Lib width = {mem_width})")
                    else:
                        print(f"    Port {port}: OK (Submodule width = {sub_width}, Top Level Connection width = {conn_width})")
                else:
                    print(f"    Port {port}: Could not determine width (Submodule width: {sub_width}, Top Level Connection: {conn})")

def validate_port_predictions(module_name, analyzer):
    """
    If a <module_name>.tcd_memory file exists, parse it to extract each port's "Function"
    (and for data ports the "Direction") and determine the expected predicted category:
      - For port CLK: expected category is "clock" if Function is "clock"
      - For port A: expected category is "address" if Function is "address"
      - For port D: expected category is "data_in" if Function is "data" and Direction is "input"
      - For port Q: expected category is "data_out" if Function is "data" and Direction is "output"
    Then, compare the expected category with the predicted category from analyzer.modules.
    """
    mem_file = os.path.join(os.getcwd(), f"{module_name}.tcd_memory")
    if not os.path.exists(mem_file):
        return  # no memory file; nothing to validate
    with open(mem_file, "r") as f:
        content = f.read()

    # Extract port blocks from the tcd_memory file.
    port_blocks = re.findall(r"(?i)port\s*\(\s*([A-Za-z0-9_]+)(\[[^\]]+\])?\s*\)\s*\{(.*?)\}", content, re.DOTALL)
    if not port_blocks:
        print(f"No port definitions found in {module_name}.tcd_memory")
        return

    expected = {}
    for port_name, dims, block in port_blocks:
        port_name = port_name.strip()
        if port_name.upper() not in {"CLK", "A", "D", "Q"}:
            continue  # we only validate these 4 signals
        func_match = re.search(r"(?i)Function\s*:\s*([^;]+);", block)
        direction_match = re.search(r"(?i)Direction\s*:\s*([^;]+);", block)
        func_val = func_match.group(1).strip().lower() if func_match else ""
        direction_val = direction_match.group(1).strip().lower() if direction_match else None
        if port_name.upper() == "CLK":
            expected[port_name.upper()] = "clock" if func_val == "clock" else "unknown"
        elif port_name.upper() == "A":
            expected[port_name.upper()] = "address" if func_val == "address" else "unknown"
        elif port_name.upper() in {"D", "Q"}:
            if func_val == "data":
                if direction_val:
                    if port_name.upper() == "D" and direction_val == "input":
                        expected[port_name.upper()] = "data_in"
                    elif port_name.upper() == "Q" and direction_val == "output":
                        expected[port_name.upper()] = "data_out"
                    else:
                        expected[port_name.upper()] = "unknown"
                else:
                    expected[port_name.upper()] = "unknown"
            else:
                expected[port_name.upper()] = "unknown"

    # Retrieve predicted categories from analyzer.
    predictions = {}
    for port, pred, conf in analyzer.modules.get(module_name, {}).get("ports", []):
        if port.upper() in {"CLK", "A", "D", "Q"}:
            predictions[port.upper()] = pred.lower()

    print(f"\nValidation of predicted port types from {module_name}.tcd_memory:")
    for port in ["CLK", "A", "D", "Q"]:
        if port in expected:
            exp_type = expected[port]
            pred_type = predictions.get(port, "not found")
            if exp_type == pred_type:
                print(f"  Port {port}: OK (Expected: {exp_type}, Predicted: {pred_type})")
            else:
                print(f"  Port {port}: MISMATCH (Expected: {exp_type}, Predicted: {pred_type})")
        else:
            print(f"  Port {port}: Not defined in memory file.")

def main():
    config_file = "block.config"
    # Get design file name from config.
    design_file = read_design_file_from_config(config_file)
    if not design_file:
        design_file = "engine.v"  # Fallback option.

    # Run design analysis using the design file.
    analyzer = run_analysis(design_file, interactive=False)

    # Read clock configuration from block.config.
    clock_names = read_clock_config(config_file)
    num_config_clocks = len(clock_names)
    
    print("\n=== Design Analysis Results ===")
    for module_name, info in analyzer.modules.items():
        print(f"\nModule: {module_name} ({info['type']})")
        print("Port Type Info:")
        for port, pred, conf in info["ports"]:
            print(f"  {port}: {pred} (Confidence: {conf:.2f})")
        
        # Filter and print bitwidth info for key port types.
        filtered = []
        for (p, pred, conf) in info["ports"]:
            if pred in ["clock", "address", "data_in", "data_out"]:
                if p in info["bitwidths"]:
                    filtered.append((p, pred, info["bitwidths"][p]))
        if filtered:
            print("\nBitwidth Info:")
            header = "  {0:25} {1:12} {2:15} {3:12}".format("Port", "Type", "Width (bits)", "Dimensions")
            print(header)
            print("  " + "-" * (len(header) - 2))
            for p, pred, bw in filtered:
                print("  {0:25} {1:12} {2:15} {3:12}".format(p, pred, bw['width'], bw['dim']))
        if info["instances"]:
            print("\nInstantiates:", ", ".join(info["instances"]))
        
        # If this is the top module, perform clock comparison.
        if info["type"].lower() == "top module":
            clock_port_count = sum(1 for port, pred, conf in info["ports"] if pred.lower() == "clock")
            print("\nClock Comparison:")
            print(f"  Clocks in block.config: {num_config_clocks}")
            print(f"  Clock ports in top module: {clock_port_count}")
            print(f"  Clocks extracted from config: {clock_names}")
            if clock_port_count == num_config_clocks:
                print("  ✅ Clock count matches the configuration.")
            else:
                print("  ❌ Clock count does not match the configuration.")
    
    # Memory comparison.
    mem_config = read_memory_config(config_file)
    mem_instantiated = extract_memory_instantiations(design_file)
    
    print("\n=== Memory Comparison ===")
    print(f"Memory modules in block.config: {mem_config}")
    print(f"Memory modules instantiated in design: {mem_instantiated}")
    
    missing_in_config = mem_instantiated - mem_config
    if missing_in_config:
        print("\nWarning: The following memory modules are instantiated in the design but do not have a defined memory type in block.config:")
        for m in missing_in_config:
            print(f"  ⚠ {m} does not have a defined memory type.")
    else:
        print("\n✅ All memory modules instantiated in the design are specified in block.config.")
    
    missing_in_design = mem_config - mem_instantiated
    if missing_in_design:
        print("\nError: The following memory modules are specified in block.config but do not exist in the design:")
        for m in missing_in_design:
            print(f"  ❌ {m}")
    else:
        print("\n✅ All memory modules specified in block.config exist in the design.")
    
    # Compare submodule bitwidths including memory library comparison (if memory file exists).
    compare_submodule_bitwidths(design_file, analyzer, config_file)
    
    # Validate predicted port types using tcd_memory files (if they exist) for each module.
    for module_name in analyzer.modules.keys():
        validate_port_predictions(module_name, analyzer)
    
    # Check memory library files for each instantiated memory module.
    check_memory_library_files(config_file, design_file)

if __name__ == "__main__":
    main()
