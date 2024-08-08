#!/usr/bin/env python

# Origin-Destination (OD) File Structure
# Pos Variable Type Explanation
# 1 w_geocode Char15 Workplace Census Block Code
# 2 h_geocode Char15 Residence Census Block Code
# 3 S000 Num Total number of jobs
# 4 SA01 Num Number of jobs of workers age 29 or younger9
# 5 SA02 Num Number of jobs for workers age 30 to 549
# 6 SA03 Num Number of jobs for workers age 55 or older9
# 7 SE01 Num Number of jobs with earnings $1250/month or less
# 8 SE02 Num Number of jobs with earnings $1251/month to $3333/month
# 9 SE03 Num Number of jobs with earnings greater than $3333/month
# 10 SI01 Num Number of jobs in Goods Producing industry sectors
# 11 SI02 Num Number of jobs in Trade, Transportation, and Utilities industry sectors
# 12 SI03 Num Number of jobs in All Other Services industry sectors
# 13 createdate Char Date on which data was created, formatted as YYYYMMDD

import sys
import pickle
import pandas as pd
import os.path
import random
import numpy as np
import functools
import time


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time for {func.__name__}: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


def perc_str(n, m):
    return "%d out of %d (%.2f%%)" % (n, m, 100.0 * float(n) / m)


@timer
def get_lodes_flows(lodes_fname):
    print("Loading", lodes_fname)
    lodes_df = pd.read_csv(lodes_fname)
    print("Loaded", len(lodes_df), "entries")

    # create a hash table mapping home geoid to destination. Use only the first 12 numbers in
    # the geoid, since those are the block group
    # The values will be probabilities, so that we can ensure all agents get allocated
    flows = {}

    flows_fname = lodes_fname + ".flows.bin"
    if os.path.exists(flows_fname):
        print("Loading flows from", flows_fname)
        with open(flows_fname, "rb") as f:
            flows= pickle.load(f)
        print("Loaded", len(flows), "home GEOID flows")
    else:
        for index, row in lodes_df.iterrows():
            h_block_group = str(row.h_geocode)[:12]
            w_block_group = str(row.w_geocode)[:12]
            n_jobs = row.S000
            if not h_block_group in flows:
                flows[h_block_group] = {}
            if not w_block_group in flows[h_block_group]:
                flows[h_block_group][w_block_group] = 0
            flows[h_block_group][w_block_group] += n_jobs
            #if index < 10:
            #    print(h_block_group, w_block_group, n_jobs)
        print("Found", len(flows), "flows")
        with open(flows_fname, "wb") as f:
            pickle.dump(flows, f)
    return flows
    

@timer
def load_urbanpop(fname):
    print("Loading", fname)
    agents_df = pd.read_csv(fname)
    print("Loaded", len(agents_df), "agents")
    num_geoids = agents_df.geoid.nunique()
    print("Number of unique GEOIDs", num_geoids)
    agents_military = len(agents_df[(agents_df.pr_emp_stat == 3)])
    agents_employed = len(agents_df[(agents_df.pr_emp_stat == 2)]) + agents_military
    print("Total employed, according to UrbanPop:", agents_employed,
          "with", perc_str(agents_military, agents_employed), "in military")
    return agents_df, agents_employed


@timer
def get_exact_work_locations(agents_df, agents_employed, flows):
    print("Getting exact work locations from LODES data")
    
    tot_flow = 0
    for h_block_group, w_block_groups in flows.items():
        tot_flow += sum(w_block_groups.values())
    print("Total employed according to LODES", tot_flow)
    
    num_agents_assigned = 0
    num_exhausted = 0
    num_not_found = 0
    num_out_of_state = 0
    missing_h_geoids = {}
    num_same_work_home = 0
    for index, agent in agents_df.iterrows():
        if agent.pr_emp_stat in [2, 3]:
            # find flows
            agent_flows = flows.get(str(agent.geoid))
            if agent_flows is None:
                num_not_found += 1
                missing_h_geoids[agent.geoid] = True
            elif not agent_flows:
                #print("agent", index, "is employed but destinations have been exhausted")
                num_exhausted += 1
            else:
                num_agents_assigned += 1
                w_geoid, num_jobs = random.choice(list(agent_flows.items()))
                if w_geoid[:2] != "35":
                    num_out_of_state += 1
                num_jobs -= 1
                if w_geoid == str(agent.geoid):
                    num_same_work_home += 1
                if num_jobs == 0:
                    del flows[str(agent.geoid)][w_geoid]
                else:
                    flows[str(agent.geoid)][w_geoid] = num_jobs


    if num_not_found > 0:
        print("WARNING: Couldn't find", perc_str(num_not_found, agents_employed),
              "agent home geoids (" + str(len(missing_h_geoids)), "unique).",
              "This likely indicates a LODES to UrbanPop data mismatch.")

    print("Set work GEOIDs for", perc_str(num_agents_assigned, agents_employed),
          "agents, with", perc_str(num_out_of_state, num_agents_assigned),
          "out of state work locations")
    print("Number of agents with same work and home locations",
          perc_str(num_same_work_home, num_agents_assigned))

    tot_leftover_flow = 0
    for h_block_group, w_block_groups in flows.items():
        for w_block_group, num_jobs in w_block_groups.items():
            tot_leftover_flow += num_jobs
    print("Number employed leftover", perc_str(tot_leftover_flow, tot_flow))


def get_w_geoid(agent, flow_probs):
    if agent.pr_emp_stat in [2, 3]:
        # find flows
        agent_flows = flow_probs.get(str(agent.geoid))
        if agent_flows is not None:
            w_geoid_idx = np.random.choice(len(agent_flows[0]), p=agent_flows[1])
            return int(agent_flows[0][w_geoid_idx])
        else:
            print("WARNING: could not find home GEOID", agent.geoid)
    return -1
    

def out_of_state(agent):
    if agent.w_geoid == -1:
        return False
    if str(agent.w_geoid)[:2] != "35":
        return True
    return False

    
@timer    
def get_prob_work_locations(agents_df, agents_employed, flows):
    print("Getting probabilistic work locations from LODES data")
    
    tot_flow = 0
    flow_probs = {}        
    for h_block_group, w_block_groups in flows.items():
        sum_vals = sum(w_block_groups.values())
        tot_flow += sum_vals
        probs = [float(x) / sum_vals for x in w_block_groups.values()]
        flow_probs[h_block_group] = [list(w_block_groups.keys()), probs]
    print("Total employed according to LODES", tot_flow)

    np.random.seed(29)
    agents_df["w_geoid"] = \
        agents_df.apply(lambda agent: get_w_geoid(agent, flow_probs), axis=1)

    num_agents_assigned = sum(agents_df.w_geoid != -1)
    num_same_work_home = sum(agents_df.w_geoid == agents_df.geoid)
    num_out_of_state = sum(agents_df.apply(lambda agent: out_of_state(agent), axis=1))

    print("Set work GEOIDs for", perc_str(num_agents_assigned, agents_employed),
          "agents, with", perc_str(num_out_of_state, num_agents_assigned),
          "out of state work locations")
    print("Number of agents with same work and home locations",
          perc_str(num_same_work_home, num_agents_assigned))

    agents_df.to_csv("agents_with_work.csv", index=True)
              

@timer
def main():
    flows = get_lodes_flows(sys.argv[1])
    # Now read in the urbanpop data csv file, and for each agent, add a work destination,
    # if appropriate
    agents_df, agents_employed = load_urbanpop(sys.argv[2])
    #agents_df = agents_df.head(100000)
    #get_exact_work_locations(agents_df, agents_employed, flows)
    get_prob_work_locations(agents_df, agents_employed, flows)

    
if __name__ == "__main__":
    main()
