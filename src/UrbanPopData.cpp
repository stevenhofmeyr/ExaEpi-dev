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

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include "UrbanPopData.H"
#include "UrbanPopAgentStruct.H"
// FIXME: this should not be included here
#include "AgentContainer.H"

using namespace amrex;

using std::string;
using std::to_string;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::ifstream;
using std::istringstream;
using std::ostringstream;

using ParallelDescriptor::MyProc;
using ParallelDescriptor::NProcs;


static vector<UrbanPopBlockGroup> read_block_groups_file(const string &fname) {
    // read in geoids file and broadcast
    Vector<char> geoids_file_ptr;
    ParallelDescriptor::ReadAndBcastFile(fname + ".geoids", geoids_file_ptr);
    string geoids_file_ptr_string(geoids_file_ptr.dataPtr());
    istringstream geoids_file_iss(geoids_file_ptr_string, istringstream::in);

    vector<UrbanPopBlockGroup> block_groups;
    UrbanPopBlockGroup block_group;
    while (true) {
        if (!block_group.read(geoids_file_iss)) break;
        block_groups.push_back(block_group);
    }
    return block_groups;
}

static void construct_geom(const string &fname) {
    auto block_groups = read_block_groups_file(fname);
    float min_lat = 1000;
    float min_long = 1000;
    float max_lat = -1000;
    float max_long = -1000;
    for (auto &block_group : block_groups) {
        max_lat = max(block_group.latitude, max_lat);
        max_long = max(block_group.longitude, max_long);
        min_lat = min(block_group.latitude, min_lat);
        min_long = min(block_group.longitude, min_long);
    }

    // grid spacing is 1/10th minute of arc at the equator, which is about 0.12 regular miles
    float gspacing = 0.1 / 60.0;
    // add a margin
    min_lat -= gspacing;
    max_lat += gspacing;
    min_long -= gspacing;
    max_long += gspacing;

    // the boundaries of the problem in real coordinates, i.e. latituted and longitude.
    RealBox real_box_latlong({AMREX_D_DECL(min_lat, min_long, 0)}, {AMREX_D_DECL(max_lat, max_long, 0)});
    // the number of grid points in a direction
    int grid_x = (max_lat - min_lat) / gspacing - 1;
    int grid_y = (max_long - min_long) / gspacing - 1;
    // the grid that overlays the domain, with the grid size in x and y directions
    Box base_domain_latlong(IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(grid_x, grid_y, 0)));
    // lat/long is a spherical coordinate system
    Geometry geom_latlong(base_domain_latlong, &real_box_latlong, CoordSys::SPHERICAL);
    // actual spacing (!= gspacing)
    float gspacing_x = geom_latlong.CellSizeArray()[0];
    float gspacing_y = geom_latlong.CellSizeArray()[1];
    Print() << "Geographic area: (" << min_lat << ", " << min_long << ") " << max_lat << ", " << max_long << ")\n";
    Print() << "Base domain: " << geom_latlong.Domain() << "\n";
    //Print() << "Geometry: " << geom_latlong << "\n";
    Print() << "Actual grid spacing: " << gspacing_x << ", "  << gspacing_y << "\n";

    // create a box array with a single box representing the domain
    BoxArray ba_latlong(geom_latlong.Domain());
    // split the box array by forcing the box size to be limited to a given number of grid points
    ba_latlong.maxSize(0.25 * grid_x / NProcs());
    Print() << "Number of boxes: " << ba_latlong.size() << "\n";
    // distribute the boxes in the array across the processors
    DistributionMapping dm_latlong;
    // weights set according to population in each box
    std::vector<Long> weights(ba_latlong.size(), 0);
    // offset into main data file for this block group
    std::vector<Long> file_offsets(ba_latlong.size(), 0);

    for (auto &block_group : block_groups) {
        // convert lat/long coords to grid coords
        int x = (block_group.latitude - min_lat) / gspacing_x;
        int y = (block_group.longitude - min_long) / gspacing_y;
        int bi_loc = -1;
        for (int bi = 0; bi < ba_latlong.size(); bi++) {
            auto bx = ba_latlong[bi];
            if (bx.contains(IntVect(x, y))) {
                bi_loc = bi;
                weights[bi] += block_group.population;
                file_offsets[bi] += block_group.file_offset;
                break;
            }
        }
        if (bi_loc == -1) AllPrint() << MyProc() << ": WARNING: could not find box for " << x << "," << y << "\n";
    }

    //dm_latlong.SFCProcessorMap(ba_latlong, weights, NProcs());
    //Print() << "ba_latlong " << ba_latlong << " dm_latlong " << dm_latlong << "\n";
    dm_latlong.KnapSackProcessorMap(weights, NProcs());
    //Print() << "ba_latlong " << ba_latlong << " dm_latlong " << dm_latlong << "\n";

    // FIXME: put this in main.cpp somehow
    AgentContainer agent_container(geom_latlong, dm_latlong, ba_latlong);

    int num_tile_boxes = 0;
    int tot_population = 0;
    for (MFIter mfi = agent_container.MakeMFIter(0, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    //for (MFIter mfi = agent_container.MakeMFIter(0); mfi.isValid(); ++mfi) {
        auto bx = mfi.tilebox();
        //auto bx = mfi.validbox();
        // find box in box array
        int bi_loc = -1;
        for (int bi = 0; bi < ba_latlong.size(); bi++) {
            //if (bx == ba_latlong[bi]) {
            if (ba_latlong[bi].contains(bx)) {
                bi_loc = bi;
                break;
            }
        }
        if (bi_loc == -1) {
            AllPrint() << MyProc() << ": WARNING: could not find box for tile box " << num_tile_boxes << "\n";
        } else {
            tot_population += weights[bi_loc];
            // for tiling - don't count multiple times
            weights[bi_loc] = 0;
        }
        num_tile_boxes++;
    }
    AllPrint() << "<" << MyProc() << ">: " << num_tile_boxes << " tile boxes, " << tot_population << " population\n";
    ParallelDescriptor::ReduceIntSum(tot_population);
    amrex::ParallelContext::BarrierAll();
    Print() << "Total population across all processors is " << tot_population << "\n";
}


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
    double load_balance = (double)all / (double)NProcs() / max_num;
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
    construct_geom(fname);
    return;

    // Each rank reads a fraction of the file.
    size_t fsize = std::filesystem::file_size(fname);
    amrex::Print() << "Size of " << fname << " is " << fsize << "\n";
    size_t chunk = fsize / NProcs();
    size_t my_offset = chunk * MyProc();


    ifstream f(fname);
    if (!f) amrex::Abort("Could not open file " + fname + "\n");
    f.seekg(my_offset);
    string buf;
    if (MyProc() == 0) {
        // the first line in the file contains the header, so skip it
        if (!getline(f, buf)) amrex::Abort("Could not read line of file " + fname + " at " + to_string(my_offset) + "\n");
    }
    int num_households = 0;
    int64_t current_geoid = -1;
    UrbanPopAgent agent;
    auto start_pos = f.tellg();
    if (MyProc() == 0) {
        agent.read_csv(f);
    } else {
        // scan until a new geoid is encountered, marking the beginning of a block group
        while (agent.read_csv(f)) {
            if (agent.p_id == -1) continue;
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
    // amrex::AllPrint() << "proc " << MyProc() << " starting at " << start_pos << " my_offset " << my_offset << "\n";
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
        if (!agent.read_csv(f)) break;
        // only keep reading until a new geoid is encountered
        if ((MyProc() < NProcs() - 1) && (stop_pos > my_offset + chunk) && (current_geoid != agent.geoid)) break;
    }

    my_num_agents = agents.size();
    // amrex::AllPrint() << "proc " << MyProc() << ": last agent " << agents[num_agents - 1].p_id << ","
    //                  << agents[num_agents - 1].h_id << "," << agents[num_agents - 1].geoid << "\n";
    // amrex::AllPrint() << "proc " << MyProc() << " stopping at " << stop_pos << " my_offset+chunk " << (my_offset + chunk) << "\n";

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(my_num_agents > 0, "Number of agents must be positive");

    amrex::ParallelContext::BarrierAll();

    p_id.resize(my_num_agents);
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
        p_id[i] = agent.p_id;
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
