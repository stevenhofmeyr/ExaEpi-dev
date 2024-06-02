/*! @file UrbanPopData.cpp
    \brief Implementation of #UrbanPopData class
*/

#include <cmath>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include "UrbanPopData.H"
#include "UrbanPopAgentStruct.H"

using namespace amrex;

using std::string;
using std::to_string;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::ifstream;
using std::ostringstream;


struct GeoidHH {
    int64_t geoid;
    int h_id;

    GeoidHH(int64_t geoid, int h_id) : geoid(geoid), h_id(h_id) {}

    bool operator==(const GeoidHH &other) const {
        return geoid == other.geoid && h_id == other.h_id;
    }
};

namespace std {
    template <>
    struct hash<GeoidHH> {
        size_t operator()(const GeoidHH &elem) const {
            return std::hash<int64_t>{}(elem.geoid) ^ (std::hash<int64_t>{}(elem.h_id) << 1);
        }
    };
}

static std::pair<int, double> get_all_load_balance(long num) {
    int all = num;
    ParallelDescriptor::ReduceIntSum(all);
    int max_num = num;
    ParallelDescriptor::ReduceIntMax(max_num);
    double load_balance = (double)all / (double)ParallelDescriptor::NProcs() / max_num;
    return {all, load_balance};
}

/*! \brief Read in UrbanPop data from given file
* Each process reads in a non-overlapping chunk of the file and uses that to populate the agent array. The reading starts at an
* offset in the file which is determined by dividing the file size by the number of processes. Hence the starting position could
* be partway through a line, and it is unlikely to be a the beginning of a block group. So each process must scan until it
* reaches a new block group.
*/
void UrbanPopData::InitFromFile (const string& fname)
{
    BL_PROFILE("UrbanPopData::InitFromFile");

    auto read_check_agent = [](UrbanPopAgent &agent, ifstream &f, const string &fname) {
        try {
            if (!agent.read_csv(f)) return false;
        } catch (const std::exception &ex) {
            ostringstream os;
            os << "Error reading file " << fname << ": " << ex.what() << "\n";
            amrex::Abort(os.str());
        }
        return true;
    };

    auto my_proc = ParallelDescriptor::MyProc();
    auto num_procs = ParallelDescriptor::NProcs();
    // Each rank reads a fraction of the file.
    size_t fsize = std::filesystem::file_size(fname);
    amrex::Print() << "Size of " << fname << " is " << fsize << "\n";
    size_t chunk = fsize / num_procs;
    size_t my_offset = chunk * my_proc;


    ifstream f(fname);
    if (!f) amrex::Abort("Could not open file " + fname + "\n");
    f.seekg(my_offset);
    string buf;
    if (my_proc == 0) {
        // the first line in the file contains the header, so skip it
        if (!getline(f, buf)) amrex::Abort("Could not read line of file " + fname + " at " + to_string(my_offset) + "\n");
    }
    int num_households = 0;
    int64_t current_geoid = -1;
    UrbanPopAgent agent;
    auto start_pos = f.tellg();
    if (my_proc == 0) {
        read_check_agent(agent, f, fname);
    } else {
        // scan until a new geoid is encountered, marking the beginning of a block group
        while (read_check_agent(agent, f, fname)) {
            if (agent.p_id == -1) continue; // incomplete line was read
            if (current_geoid == -1) current_geoid = agent.geoid;
            if (current_geoid != agent.geoid) break;
            start_pos = f.tellg();
        }
    }
    // each process will read a unique, non-overlapping, set of agents
    vector<UrbanPopAgent> agents;
    // used for counting up the number of unique census block groups
    unordered_set<int64_t> unique_geoids;
    // used for counting up the number of unique households and assigning a unique number to each
    // note that the h_id is only unique to a block group, and not across block groups. Hence the hash table uses the geoid, h_id
    unordered_map<GeoidHH, int> households;
    // now read in agents starting from the new block group
    // amrex::AllPrint() << "proc " << my_proc << " starting at " << start_pos << " my_offset " << my_offset << "\n";
    auto stop_pos = f.tellg();
    while (true) {
        // the agent is already set to the first one for this process, so we first add it
        agents.push_back(agent);
        unique_geoids.insert(agent.geoid);
        auto elem = households.find({agent.geoid, agent.h_id});
        if (elem == households.end()) households.insert({GeoidHH(agent.geoid, agent.h_id), num_households++});
        current_geoid = agent.geoid;
        // now read the next agent
        stop_pos = f.tellg();
        if (!read_check_agent(agent, f, fname)) break;
        // only keep reading until a new geoid is encountered
        if ((my_proc < num_procs - 1) && (stop_pos > my_offset + chunk) && (current_geoid != agent.geoid)) break;
    }

    my_num_agents = agents.size();
    // amrex::AllPrint() << "proc " << my_proc << ": last agent " << agents[num_agents - 1].p_id << ","
    //                  << agents[num_agents - 1].h_id << "," << agents[num_agents - 1].geoid << "\n";
    // amrex::AllPrint() << "proc " << my_proc << " stopping at " << stop_pos << " my_offset+chunk " << (my_offset + chunk) << "\n";

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(my_num_agents > 0, "Number of agents must be positive");

    amrex::ParallelContext::BarrierAll();

    h_id.resize(my_num_agents);
    geoid.resize(my_num_agents);
    pr_age.resize(my_num_agents);
    pr_emp_stat.resize(my_num_agents);
    pr_commute.resize(my_num_agents);

    int num_employed = 0;
    int num_military = 0;
    for (int i = 0; i < my_num_agents; i++) {
        UrbanPopAgent &agent = agents[i];
        geoid[i] = agent.geoid;
        h_id[i] = households[GeoidHH(agent.geoid, agent.h_id)];
        pr_age[i] = agent.pr_age;
        pr_emp_stat[i] = agent.pr_emp_stat;
        // crude estimate based on employment status
        if (agent.pr_emp_stat == 2) num_employed++;
        if (agent.pr_emp_stat == 3) num_military++;
        pr_commute[i] = agent.pr_commute;
    }

    my_num_block_groups = unique_geoids.size();
    auto [all_num_block_groups, load_balance_block_groups] = get_all_load_balance(my_num_block_groups);
    this->all_num_block_groups = all_num_block_groups;
    auto [all_num_agents, load_balance_agents] = get_all_load_balance(my_num_agents);
    this->all_num_agents = all_num_agents;
    ParallelDescriptor::ReduceIntSum(num_employed);
    ParallelDescriptor::ReduceIntSum(num_military);
    ParallelDescriptor::ReduceIntSum(num_households);

    amrex::Print() << "Population:  " << all_num_agents << " (balance "
                                      << std::fixed << std::setprecision(3) << load_balance_agents << ")\n";
    amrex::Print() << "Employed:    " << num_employed << "\n";
    amrex::Print() << "Military:    " << num_military << "\n";
    amrex::Print() << "Households:  " << num_households << "\n";
    amrex::Print() << "Communities: " << all_num_block_groups << " (balance "
                                      << std::fixed << std::setprecision(3) << load_balance_block_groups << ")\n";

    CopyDataToDevice();
    amrex::Gpu::streamSynchronize();
}

/*! \brief Prints UrbanPop data to screen */
void UrbanPopData::Print () const {
    amrex::Print() << my_num_agents << "\n";
    for (int i = 0; i < my_num_agents; ++i) {
        amrex::Print() << i << " " << geoid[i] << " " << h_id[i] << " " << pr_age[i] << " " << pr_emp_stat[i] << " "
                       << pr_commute[i] << "\n";
    }
}

/*! \brief Copy array from host to device */
template<typename T>
void UrbanPopData::CopyToDeviceAsync (const amrex::Vector<T>& h_vec, /*!< host vector */
                                      amrex::Gpu::DeviceVector<T>& d_vec /*!< device vector */) {
    d_vec.resize(0);
    d_vec.resize(h_vec.size());
    Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
}

/*! \brief Copy array from device to host */
template<typename T>
void UrbanPopData::CopyToHostAsync (const amrex::Gpu::DeviceVector<T>& d_vec, /*!< device vector */
                                    amrex::Vector<T>& h_vec /*!< host vector */) {
    h_vec.resize(0);
    h_vec.resize(d_vec.size());
    Gpu::copyAsync(Gpu::deviceToHost, d_vec.begin(), d_vec.end(), h_vec.begin());
}

/*! \brief Copies member arrays of #UrbanPopData from host to device */
void UrbanPopData::CopyDataToDevice () {
    CopyToDeviceAsync<int64_t>(geoid, geoid_d);
    CopyToDeviceAsync<int>(h_id, h_id_d);
    CopyToDeviceAsync<int>(pr_age, pr_age_d);
    CopyToDeviceAsync<int>(pr_emp_stat, pr_emp_stat_d);
    CopyToDeviceAsync<int>(pr_commute, pr_commute_d);
}
