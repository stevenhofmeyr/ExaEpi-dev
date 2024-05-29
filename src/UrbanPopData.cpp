/*! @file UrbanPopData.cpp
    \brief Implementation of #UrbanPopData class
*/

#include <cmath>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include "UrbanPopData.H"
#include "UrbanPopAgentStruct.H"

using namespace amrex;

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

/*! \brief Read in UrbanPop data from given file */
void UrbanPopData::InitFromFile (const std::string& fname)
{
    BL_PROFILE("UrbanPopData::InitFromFile");

    // FIXME: currently, each rank reads in the whole file. This is very inefficient for larger files and we expect the full
    // US-scale UrbanPop data to be around 40GB. Hence each rank should read in a subset of the file and use that to populate
    // its data structures

    std::vector<UrbanPopAgent> agents;
    std::ifstream f(fname);
    if (!f) amrex::Abort("Could not open file " + fname + "\n");
    // the first line contains the header
    std::string buf;
    if (!getline(f, buf)) amrex::Abort("Could not read first line of file " + fname + "\n");
    int line = 0;
    for (;; line++) {
        UrbanPopAgent agent;
        try {
            if (!agent.read_csv(f)) break;
        } catch (const std::exception &ex) {
            std::ostringstream os;
            os << "Error reading file " << fname << " on line " << line << ": " << ex.what() << std::endl;
            amrex::Abort(os.str());
        }
        agents.push_back(agent);
    }

    num_agents = agents.size();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(num_agents > 0, "Number of agents must be positive");

    h_id.resize(num_agents);
    geoid.resize(num_agents);
    pr_age.resize(num_agents);
    pr_emp_stat.resize(num_agents);
    pr_commute.resize(num_agents);

    int num_employed = 0;
    int num_workers = 0;
    int num_military = 0;
    // used for counting up the number of unique census block groups
    std::unordered_set<int64_t> unique_geoids;
    // used for counting up the number of unique households and the number working in each household
    // note that the h_id is only unique to a block group, and not across block groups. Hence the hash table uses the geoid, h_id
    // as the key
    std::unordered_map<GeoidHH, int> households;
    for (int i = 0; i < num_agents; i++) {
        UrbanPopAgent &agent = agents[i];
        geoid[i] = agent.geoid;
        h_id[i] = agent.h_id;
        pr_age[i] = agent.pr_age;
        pr_emp_stat[i] = agent.pr_emp_stat;
        // crude estimate based on employment status
        //if (agent.pr_emp_stat == 2 || agent.pr_emp_stat == 3) num_employed++;
        if (agent.pr_emp_stat == 2) num_employed++;
        if (agent.pr_emp_stat == 3) num_military++;
        pr_commute[i] = agent.pr_commute;
        unique_geoids.insert(agent.geoid);
        auto elem = households.find({agent.geoid, agent.h_id});
        if (elem == households.end()) {
            households.insert({GeoidHH(agent.geoid, agent.h_id), agent.hh_nb_wrks});
            num_workers += agent.hh_nb_wrks;
        } else if (elem->second != agent.hh_nb_wrks) {
            std::cerr << "agent mismatch in nb_wrks: previously " << elem->second << ", now " << (int)agent.hh_nb_wrks << "\n";
        }
    }

    num_block_groups = unique_geoids.size();

    amrex::Print() << "Total population " << num_agents << "\n";
    amrex::Print() << "Total employed " << num_employed << "\n";
    amrex::Print() << "Total military " << num_military << "\n";
    amrex::Print() << "Total workers " << num_workers << "\n";
    amrex::Print() << "Number of households: " << households.size() << "\n";
    amrex::Print() << "Number of communities: " << num_block_groups << "\n";

    CopyDataToDevice();
    amrex::Gpu::streamSynchronize();
}

/*! \brief Prints UrbanPop data to screen */
void UrbanPopData::Print () const {
    amrex::Print() << num_agents << "\n";
    for (int i = 0; i < num_agents; ++i) {
        amrex::Print() << i << " " << geoid[i] << " " << h_id[i] << " " << pr_age[i] << " "
                       << pr_emp_stat[i] << " " << pr_commute[i] << "\n";
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
