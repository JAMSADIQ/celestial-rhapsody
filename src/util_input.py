"""Module for reading and processing config file for Planetary Motions problem
"""


def read_config(file_name):
    """Function for reading and processing config file

    Args:
        file_name (string): path and file name for config file

    Raises:
        RuntimeError: raises error if failed to open and process config file

    Returns:
        list, list of list: returns a list that contains global variables,
        and a list of list that contains planet variables.
    """
    global_var = []
    planet_var = []
    allplanet_var = []

    planet_flag = False

    try:
        with open(file_name, "r") as file:
            line = None
            prev = next(file, None)

            for line in file:
                """Append non empty lines to global_var until finding non empty lines
                that have 'planet' keyword. Append next non empty lines to planet_var list.
                If finding lines that have another 'planet' keyword or is the end of file,
                append planet_var list to allplanet_var list.
                """
                if line.strip():
                    if "Planet" in line:
                        planet_flag = True
                        if bool(planet_var):
                            allplanet_var.append(planet_var[:])
                            planet_var.clear()
                        continue

                    if not planet_flag:
                        if "final_time" in line:
                            global_var.append(int(line.split("=", 1)[1].strip()))
                        else:
                            global_var.append(float(line.split("=", 1)[1].strip()))
                    else:
                        try:
                            planet_var.append(float(line.split("=", 1)[1].strip()))
                        except:
                            planet_var.append(line.split("=", 1)[1].strip())
                prev = line

            if prev is not None:
                if bool(planet_var):
                    allplanet_var.append(planet_var[:])
                    planet_var.clear()
    except:
        raise RuntimeError(
            "Failed to process config file, make sure to input the correct config file."
        )

    return global_var, allplanet_var


def read_config_alt(file_name):
    """Alternative function for reading and processing config file

    Args:
        file_name (string): path and file name for config file

    Raises:
        RuntimeError: raises error if failed to open and process config file

    Returns:
        list, list of list: returns a list that contains global variables,
        and a list of list that contains planet variables.
    """
    lines = []
    global_var = []
    planet_var = []
    allplanet_var = []

    try:
        with open(file_name, "r") as file:
            for line in file:
                if line.strip():
                    lines.append(line.strip())
    except:
        raise RuntimeError(
            "Failed to process config file, make sure to input the correct config file."
        )

    for line in lines:
        if "Global" in line:
            planet_flag = False
            continue
        elif "Planet" in line:
            planet_flag = True
            if bool(planet_var):
                allplanet_var.append(planet_var[:])
                planet_var.clear()
            continue

        if not planet_flag:
            global_var.append(float(line.split("=", 1)[1].strip()))
        else:
            try:
                planet_var.append(float(line.split("=", 1)[1].strip()))
            except:
                planet_var.append(line.split("=", 1)[1].strip())

    if bool(planet_var):
        allplanet_var.append(planet_var[:])
        planet_var.clear()

    return global_var, allplanet_var


def main():
    global_var, allplanet_var = read_config("input/config_test.txt")
    print(f"Global variable is {global_var}")
    print(f"Planets variable are {allplanet_var}")

    global_var, allplanet_var = read_config_alt("input/config_test.txt")
    print(f"Global variable is {global_var}")
    print(f"Planets variable are {allplanet_var}")


if __name__ == "__main__":
    main()
