#!/usr/bin/env python

import os.path
import pandas
from pandas.api.types import CategoricalDtype
import numpy as np
import time
import argparse
import geopandas


PUMS_ID_LEN = 14
NAICS_LEN = 9


# note: missing category indexes are written as -1
# we could extract these from the dataset, but then they will not be in a suitable order, even with sorting
# so we predefine and check for values in the data that are not listed here
categ_types = {
    'hh_type':
        CategoricalDtype(categories=["hh", "gq"]),
    'hh_living_arrangement':
        CategoricalDtype(categories=["married", "male_no_spouse", "female_no_spouse", "alone", "not_alone"]),
    'hh_has_kids':
        CategoricalDtype(categories=["no", "yes"]),
    'hh_dwg':
        CategoricalDtype(categories=[
            "single_fam_detach", "single_fam_attach", "2_unit", "3_4_unit", "5_9_unit", "10_19_unit", "20_49_unit", "GE50_unit",
            "mob_home", "other"]),
    'hh_tenure':
        CategoricalDtype(categories=["own", "rent", "other"]),
    'hh_vehicles':
        CategoricalDtype(categories=["01", "02", "03", "04", "05", "GE06"]),
    'pr_sex':
        CategoricalDtype(categories=["female", "male"]),
    'pr_race':
        CategoricalDtype(categories=["white", "blk_af_amer", "asian", "native_amer", "pac_island", "other", "mult"]),
    'pr_hsplat':
        CategoricalDtype(categories=["no", "yes"]),
    'pr_ipr':
        CategoricalDtype(categories=["L050", "050_099", "100_124", "125_149", "150_184", "185_199", "GE200"]),
    'pr_emp_stat':
        CategoricalDtype(categories=["not.in.force", "unemp", "employed", "mil"]),
    'pr_travel':
        CategoricalDtype(
            categories=["car_truck_van", "public_transportation", "bicycle", "walked", "motorcycle", "taxicab", "other", "wfh"]),
    'pr_veh_occ':
        CategoricalDtype(categories=["drove_alone", "carpooled"]),
    'pr_grade':
        CategoricalDtype(categories=[
            "preschl", "kind", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
            "undergrad", "grad"])
}



def print_header(df):
    string_fields = {"pums_id": "PUMS_ID_LEN", "pr_naics": "NAICS_LEN"}

    hdr_fname = "UrbanPopAgentStruct.H"
    print("Writing C++ header file at", hdr_fname)
    # print out header file for C++ import
    hdr = "#include <stdlib.h>\n" \
          "#include <string.h>\n" \
          "#include <fstream>\n" \
          "#include <sstream>\n\n" \
          "using float32_t = float;\n\n" \
          "const size_t PUMS_ID_LEN = " + str(PUMS_ID_LEN) + ";\n" \
          "const size_t NAICS_LEN = " + str(NAICS_LEN) + ";\n\n"

    # print out string arrays with category names
    for field_type in df:
        if field_type in categ_types:
            categs_expected = list(categ_types[field_type].categories)
            hdr += 'static std::vector<std::string> ' + field_type + '_descriptions = {"' + '", "'.join(categs_expected) + '"};\n'
    hdr += '\n'

    hdr += 'struct UrbanPopAgent {\n\n'
    for i, col in enumerate(df.columns):
        if col in string_fields:
            hdr += '    char ' + col + '[' + string_fields[col] + '];\n'
        else:
            hdr += '    ' + str(df.dtypes.iloc[i]) + '_t ' + col + ';\n'

    hdr += """
    bool read_csv(std::ifstream &f) {
        auto split = [](const std::string &s, char delim) {
            std::vector<std::string> elems;
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, delim)) elems.push_back(item);
            return elems;
        };

        std::string buf;
        if (!getline(f, buf)) return false;
        auto tokens = split(buf, ',');\n\n"""

    for i, col in enumerate(df.columns):
        if col in string_fields:
            hdr += "        if (!tokens[" + str(i) + "].empty()) " \
                   "strncpy(" + col + ", tokens[" + str(i) + "].c_str(), " + string_fields[col] + ");\n" \
                   "        else memset(" + col + ", 0, " + string_fields[col] + ");\n"
        else:
            hdr += "        " + col + " = stof(tokens[" + str(i) + "]);\n"

    hdr += """
        return true;
    }\n"""

    hdr += """
    friend std::ostream& operator<<(std::ostream& os, const UrbanPopAgent& agent) {\n"""

    for i, col in enumerate(df.columns):
        c_type = str(df.dtypes.iloc[i]) + "_t"
        if col in string_fields:
            hdr += '        os << std::string(agent.' + col + ', ' + string_fields[col] + ') << ",";\n'
        elif col in categ_types:
            hdr += '        os << (int)agent.' + col + ' << (agent.' + col + ' != -1 ? ":" + ' + col + '_descriptions[agent.' + \
                   col + '] : "") << ",";\n'
        else:
            hdr += "        os << " + ("(int)" if c_type == "int8_t" else "") + "agent." + col + " << ',';\n"

        #os << (int)agent.hh_dwg << (agent.hh_dwg != -1 ? ":" + hh_dwg_descriptions[agent.hh_dwg] : "") << ",";

    hdr += """
        return os;
    }\n"""

    hdr += "};"

    f_hdr = open(hdr_fname, "w")
    print(hdr, file=f_hdr)
    f_hdr.close()
    print("Wrote", len(df.columns), "fields to", hdr_fname)


def process_feather_files(fnames, out_fname, geoid_locs_map):
    global PUMS_ID_LEN
    global NAICS_LEN

    dfs = []
    for fname in fnames:
        print("Reading data from", fname)
        t = time.time()
        dfs.append(pandas.read_feather(fname))
        print("Read", len(dfs[-1].index), "records in %.3f s" % (time.time() - t))

    df = pandas.concat(dfs)
    # set specific ID types
    df.p_id = df.p_id.str.split("-").str[-1].astype("int32")
    PUMS_ID_LEN = df.pums_id.map(len).max()
    df.h_id = df.h_id.str.split("-").str[-1].astype("int32")
    df.geoid = df.geoid.astype("int64")
    NAICS_LEN = df.pr_naics.map(len).max()
    print("Unique PUMS", len(df.pums_id.unique()), "max length", PUMS_ID_LEN)
    print("Unique NAICS", len(df.pr_naics.unique()), "max length", NAICS_LEN)
    # pack structure by moving int32_t value to before all int8_t values
    df.insert(4, "hh_income", df.pop("hh_income"))
    # move char arrays to end of struct
    df.insert(len(df.columns) - 1, "pums_id", df.pop("pums_id"))
    df.insert(len(df.columns) - 1, "pr_naics", df.pop("pr_naics"))

    print("Categorical types")
    for field_type in df:
        if field_type in categ_types:
            # compare the list with the unique to make sure we haven't fonud anything not in our predefined list
            categs_found = list(df[field_type].unique())
            categs_expected = list(categ_types[field_type].categories)
            missing = [x for x in categs_found if x not in categs_expected and x != '' and x is not None]
            if missing:
                print("WARNING: Found missing categories for", field_type, ":", missing)
            df[field_type] = df[field_type].astype(categ_types[field_type]).cat.codes
        else:
            if df.dtypes[field_type] == object:
                df[field_type] = df[field_type].astype("string")
            else:
                # this converts all float to int
                max_val = df[field_type].max()
                if max_val < 128:
                    df[field_type] = df[field_type].astype("int8")
                elif max_val < 2**15:
                    df[field_type] = df[field_type].astype("int16")
                elif max_val < 2**31:
                    df[field_type] = df[field_type].astype("int32")
                else:
                    df[field_type] = df[field_type].astype("int64")

    if geoid_locs_map:
        # add lat/long locations from geoids
        df.insert(df.columns.get_loc("geoid") + 1, "latitude", float(0))
        df.latitude = df.latitude.astype("float32")
        df.insert(df.columns.get_loc("latitude") + 1, "longitude", float(0))
        df.longitude = df.longitude.astype("float32")

    print_header(df)

    t = time.time()

    if geoid_locs_map:
        print("Setting lat/long for data", end=" ", flush=True)
        # find lat/long for each row entry
        df["latitude"] = df["geoid"].map(geoid_locs_map).apply(lambda x: x[0]).astype("float32")
        df["longitude"] = df["geoid"].map(geoid_locs_map).apply(lambda x: x[1]).astype("float32")
        print("\nSet lat/long for", len(df.index), "agents in %.3f s" % (time.time() - t))

    print("Fields are:\n", df.dtypes, sep="")

    print("Writing CSV text data to", out_fname)
    t = time.time()
    df.to_csv(out_fname, index=False)
    print("Wrote", len(df.index), "records in %.3f s" % (time.time() - t))



def process_census_bg_shape_file(dir_names, geoid_locs_map):
    for dname in dir_names:
        # remove trailing slash if it exists
        if dname[-1] == "/":
            dname = dname[:-1]
        shape_fname = dname + "/" + os.path.split(dname)[1] + ".shp"
        # don't actually need to compute the centroid because the block group file has it under the INTPTLAT10 and INTPTLON10 cols
        #df = geopandas.read_file(shape_fname, include_fields=["GEOID10", "geometry"])
        #df = df.to_crs(crs=4326)
        #df["centroid"] = df.centroid
        print("Reading shape files at", shape_fname)
        df = geopandas.read_file(shape_fname, include_fields=["GEOID10", "INTPTLAT10", "INTPTLON10"], ignore_geometry=True)
        df.GEOID10 = df.GEOID10.astype("int64")
        df.INTPTLAT10 = df.INTPTLAT10.astype("float32")
        df.INTPTLON10 = df.INTPTLON10.astype("float32")
        df.to_csv(shape_fname + ".csv")
        print("Wrote", len(df.index), "GEOID locations to", shape_fname + ".csv")
        #print(df.dtypes)
        geoid_locs_map.update(df.set_index("GEOID10").T.to_dict("list"))


if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser(description="Convert UrbanPop feather files to C++ struct binary file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--files", "-f", required=True, nargs="+", help="Feather files")
    parser.add_argument("--shape_files_dir", "-s", nargs="+",
                        help="Directories for census block group shape files. Available from\n" + \
                        "https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2010&layergroup=Block+Groups")
    args = parser.parse_args()

    geoid_locs_map = {}
    if args.shape_files_dir is not None:
        process_census_bg_shape_file(args.shape_files_dir, geoid_locs_map)
        print("GEOID to locations map contains", len(geoid_locs_map), "entries")

    process_feather_files(args.files, args.output, geoid_locs_map)

    print("Processed", len(args.files), "files in %.2f s" % (time.time() - t))
