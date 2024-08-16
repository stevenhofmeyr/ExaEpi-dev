#!/usr/bin/env python

import sys
import os.path
import os
import pandas
from pandas.api.types import CategoricalDtype
import numpy as np
import time
import argparse
import geopandas
import scipy
import scipy.stats
import process_lodes7


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

    hdr = f"""
/*! @file {hdr_fname}
    \\brief Contains #UrbanPopAgent class used for reading in UrbanPop data
    File automatically generated by UrbanPop-scripts/extract_urbanpop_feather.py
*/

#pragma once

#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sstream>
using std::string;
using float32_t = float;

const size_t PUMS_ID_LEN = {PUMS_ID_LEN};
const size_t NAICS_LEN = {NAICS_LEN};
const size_t NUM_COLS = {len(df.columns)};

"""

    # print out string arrays with category names
    for field_type in df:
        if field_type in categ_types:
            categs_expected = list(categ_types[field_type].categories)
            hdr += f"""static string {field_type}_descriptions[] = {{"{'", "'.join(categs_expected)}"}};\n"""
    hdr += "\n"

    hdr += """
static std::vector<string> split_string(const string &s, char delim) {
    std::vector<string> elems;
    std::stringstream ss(s);
    string item;
    while (std::getline(ss, item, delim)) elems.push_back(item);
    return elems;
}

"""

    hdr += 'struct UrbanPopAgent {\n'
    for i, col in enumerate(df.columns):
        if col in string_fields:
            hdr += f"""    char {col}[{string_fields[col]}];\n"""
        else:
            hdr += f"""    {df.dtypes.iloc[i]}_t {col};\n"""

    hdr += """
    bool read_csv(std::ifstream &f) {
        string buf;
        if (!getline(f, buf)) return false;
        if (buf[0] != '*') {
            p_id = -1;
            return true;
        }
        try {
            std::vector<string> tokens = split_string(buf.substr(2), ',');
            if (tokens.size() != NUM_COLS)
                throw std::runtime_error("Incorrect number of tokens, expected " + std::to_string(NUM_COLS) +
                                         " got " + std::to_string(tokens.size()));\n"""

    for i, col in enumerate(df.columns):
        if col in string_fields:
            hdr += f"""            AMREX_ALWAYS_ASSERT(!tokens[{i}].empty());\n"""
            hdr += f"""            strncpy({col}, tokens[{i}].c_str(), {string_fields[col]});\n"""
        else:
            if df.dtypes.iloc[i] == "float32":
                hdr += f"""            {col} = stof(tokens[{i}]);\n"""
            elif df.dtypes.iloc[i] == "int64":
                hdr += f"""            {col} = stol(tokens[{i}]);\n"""
            else:
                hdr += f"""            {col} = stoi(tokens[{i}]);\n"""
    hdr += """
        } catch (const std::exception &ex) {
            std::ostringstream os;
            os << "Error reading UrbanPop input file: " << ex.what() << ", line read: " << "'" << buf << "'";
            amrex::Abort(os.str());
        }
        return true;
    }\n"""

    hdr += """
    friend std::ostream& operator<<(std::ostream& os, const UrbanPopAgent& agent) {\n"""

    for i, col in enumerate(df.columns):
        c_type = str(df.dtypes.iloc[i]) + "_t"
        if col in string_fields:
            hdr += '        os << string(agent.' + col + ', ' + string_fields[col] + ') << ",";\n'
        elif col in categ_types:
            hdr += '        os << (int)agent.' + col + ' << (agent.' + col + ' != -1 ? ":" + ' + col + '_descriptions[agent.' + \
                   col + '] : "") << ",";\n'
        else:
            hdr += "        os << " + ("(int)" if c_type == "int8_t" else "") + "agent." + col + " << ',';\n"

        #os << (int)agent.hh_dwg << (agent.hh_dwg != -1 ? ":" + hh_dwg_descriptions[agent.hh_dwg] : "") << ",";

    hdr += """
        return os;
    }
};
"""

    f_hdr = open(hdr_fname, "w")
    print(hdr, file=f_hdr)
    f_hdr.close()
    print("Wrote", len(df.columns), "fields to", hdr_fname)


def process_feather_files(fnames, out_fname, geoid_locs_map, lodes_flows):
    global PUMS_ID_LEN
    global NAICS_LEN

    dfs = []
    for fname in fnames:
        print("Reading data from", fname, end=': ')
        t = time.time()
        dfs.append(pandas.read_feather(fname))
        print(len(dfs[-1].index), "records in %.3f s" % (time.time() - t))

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
    # ensure the NAICS fields don't contain an empty string, which can muddle parsing down the line
    #df['pr_naics'] = df['pr_naics'].replace(["^\s*$"], 'NA', regex=True)
    df['pr_naics'] = df['pr_naics'].replace([''], 'NA', regex=True)

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

    df.rename(columns={"geoid": "h_geoid"}, inplace=True)

    if geoid_locs_map:
        # add lat/long locations from geoids
        df.insert(df.columns.get_loc("h_geoid") + 1, "h_lat", float(0))
        df.h_lat = df.h_lat.astype("float32")
        df.insert(df.columns.get_loc("h_lat") + 1, "h_long", float(0))
        df.h_long = df.h_long.astype("float32")

    if lodes_flows:
        df.insert(df.columns.get_loc("h_long") + 1, "w_geoid", float(-1))
        df.w_geoid = df.w_geoid.astype("int64")
        df.insert(df.columns.get_loc("w_geoid") + 1, "w_lat", float(0))
        df.w_lat = df.w_lat.astype("float32")
        df.insert(df.columns.get_loc("w_lat") + 1, "w_long", float(0))
        df.w_long = df.w_long.astype("float32")


    print_header(df)

    t = time.time()

    if geoid_locs_map:
        print("Setting lat/long for data", end=" ", flush=True)
        # find lat/long for each row entry
        df["h_lat"] = df["h_geoid"].map(geoid_locs_map).apply(lambda x: x[0]).astype("float32")
        df["h_long"] = df["h_geoid"].map(geoid_locs_map).apply(lambda x: x[1]).astype("float32")
        print("\nSet lat/long for", len(df.index), "agents in %.3f s" % (time.time() - t))

    if lodes_flows:
        print("Setting work GEOIDs for data")
        # first get exact work locations
        #df["w_geoid"] = process_lodes7.get_work_locations(df, lodes_flows, use_prob=False)
        # now get probabalisitically for the ones that weren't allocated
        df["w_geoid"] = process_lodes7.get_work_locations(df, lodes_flows, use_prob=True)
        df["w_lat"] = df["w_geoid"].map(geoid_locs_map).apply(lambda x: -1 if isinstance(x, float) else x[0]).astype("float32")
        df["w_long"] = df["w_geoid"].map(geoid_locs_map).apply(lambda x: -1 if isinstance(x, float) else x[1]).astype("float32")

        df_workerflows = df.groupby(['h_geoid', 'w_geoid']).size()
        generated_workerflows = {}
        #df_workerflows.to_csv(out_fname + "-generated_workerflows")
        for (h_geoid, w_geoid), counts in df_workerflows.items():
            if w_geoid == -1:
                continue
            if not h_geoid in generated_workerflows:
                generated_workerflows[h_geoid] = {}
            generated_workerflows[h_geoid][w_geoid] = counts

        lodes_counts = []
        generated_counts = []
        for h_block_group, w_block_groups in lodes_flows.items():
            h_block_group = int(h_block_group)
            for w_block_group, counts in w_block_groups.items():
                w_block_group = int(w_block_group)
                lodes_counts.append(counts)
                if h_block_group in generated_workerflows and w_block_group in generated_workerflows[h_block_group]:
                    generated_counts.append(generated_workerflows[h_block_group][w_block_group])
                else:
                    generated_counts.append(0)
        p = np.array(lodes_counts)
        q = np.array(generated_counts)
        d = abs(p - q)
        print("Mean diffs", np.mean(d))
        print("Total LODES flows", p.sum(), "total generated flows", q.sum())
        print("Wasserstein distance %.5f, max LODES %.0f, max generated %.0f" % \
              (scipy.stats.wasserstein_distance(p, q), max(p), max(q)))
        p = p / p.sum()
        q = q / q.sum()
        m = (p + q) / 2
        divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
        print("Jenson-Shannon divergence between LODES and generated: %.3f (%d items)" % (np.sqrt(divergence), len(lodes_counts)))
        if False:
            p_hist, _ = np.histogram(p, bins=1000, density=True)
            q_hist, _ = np.histogram(q, bins=1000, density=True)
            p_hist = p_hist / p_hist.sum()
            q_hist = q_hist / q_hist.sum()
            m_hist = (p_hist + q_hist) / 2
            divergence = (scipy.stats.entropy(p_hist, m_hist) + scipy.stats.entropy(q_hist, m_hist)) / 2
            print("Jenson-Shannon divergence (histogram) between LODES and generated: %.3f (%d items)" %
                (np.sqrt(divergence), len(lodes_counts)))
            print("Wasserstein distance (histogram) %.5f, max value %.5f" % (scipy.stats.wasserstein_distance(p_hist, q_hist),
                                                                max(max(p_hist), max(q_hist))))

    print("Fields are:\n", df.dtypes, sep="")

    #print("Sorting by geoid")
    print("Sorting by lat/long")
    t = time.time()
    df.sort_values(by=["h_lat", "h_long"], inplace=True)
    print("Sorted in %.3f s" % (time.time() - t))
    print("Writing CSV text data to", out_fname, "and block group summaries to", out_fname + ".geoids")
    t = time.time()
    num_rows = len(df.index)
    # make sure all the p_ids are globally unique (they are only unique to each urbanpop feather file originally)
    df['p_id'] = np.arange(0, num_rows)
    # start with a distinct marker so that the file can be read in parallel more easily
    df.index = ['*'] * num_rows

    w_geoids_map = {}
    w_geoids = df.w_geoid.unique()
    for i, geoid in enumerate(w_geoids):
        subset_df = df.loc[df['w_geoid'] == geoid]
        w_geoids_map[geoid] = len(subset_df.index)
    print("Found", len(w_geoids_map), "work locations")

    # print each geoid in turn so we can track the file offsets
    with open(out_fname + ".geoids", mode='w') as f:
        print("geoid lat lng foff h_pop w_pop", file=f)
        geoids = df.h_geoid.unique()
        for i, geoid in enumerate(geoids):
            foffset = os.stat(out_fname).st_size if i > 0 else 0
            subset_df = df.loc[df['h_geoid'] == geoid]
            subset_df.to_csv(out_fname, index=True, header=(i == 0), mode='w' if i == 0 else 'a')
            w_pop = w_geoids_map[geoid] if geoid in w_geoids_map else 0
            print(geoid, ' '.join(map(str, geoid_locs_map[geoid])), foffset, len(subset_df.index), w_pop, file=f)
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
    parser.add_argument("--lodes", "-l", help="LODES7 file")
    args = parser.parse_args()

    geoid_locs_map = {}
    if args.shape_files_dir is not None:
        process_census_bg_shape_file(args.shape_files_dir, geoid_locs_map)
        print("GEOID to locations map contains", len(geoid_locs_map), "entries")

    lodes_flows = {}
    if args.lodes is not None:
        lodes_flows = process_lodes7.get_lodes_flows(args.lodes)
        print("Obtained workerflow data from LODES7 file", args.lodes)

    process_feather_files(args.files, args.output, geoid_locs_map, lodes_flows)

    print("Processed", len(args.files), "files in %.2f s" % (time.time() - t))
