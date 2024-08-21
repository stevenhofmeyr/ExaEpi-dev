/*! @file AgentContainer.cpp
    \brief Function implementations for #AgentContainer class
*/

#include "AgentContainer.H"

using namespace amrex;

using std::string;
using std::to_string;

using ParallelDescriptor::MyProc;
using ParallelDescriptor::NProcs;

/*! \brief Assigns school by taking a random number between 0 and 100, and using
 *  default distribution to choose elementary/middle/high school. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int assign_school (const int nborhood, const amrex::RandomEngine& engine) {
    int il4 = amrex::Random_int(100, engine);
    int school = -1;

    if (il4 < 36) {
        school = 3 + (nborhood / 2);  /* elementary school */
    }
    else if (il4 < 68) {
        school = 2;  /* middle school */
    }

    else if (il4 < 93) {
        school = 1;  /* high school */
    }
    else {
        school = 0;  /* not in school, presumably 18-year-olds or some home-schooled */
    }
    return school;
}

/*! \brief Shuffle the elements of a given vector */
void randomShuffle (std::vector<int>& vec /*!< Vector to be shuffled */)
{
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(vec.begin(), vec.end(), g);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void set_particle_pos(Real &p1, Real &p2, int x, int y, Real dx, Real dy, short _ic_type, Real min_pos_x, Real min_pos_y) {
    if (_ic_type == ExaEpi::ICType::Census) {
        p1 = ((Real)x + 0.5_rt) * dx;
        p2 = ((Real)y + 0.5_rt) * dy;
    } else if (_ic_type == ExaEpi::ICType::UrbanPop) {
        p1 = (Real)x * dx + min_pos_x;
        p2 = (Real)y * dy + min_pos_y;
    } else {
        Abort("ic_type not supported");
    }
}

/*! \brief Initialize agents for ExaEpi::ICType::Census
 *  + Define and allocate the following integer MultiFabs:
 *    + num_families: number of families; has 7 components, each component is the
 *      number of families of size (component+1)
 *    + fam_offsets: offset array for each family (i.e., each component of each grid cell), where the
 *      offset is the total number of people before this family while iterating over the grid.
 *    + fam_id: ID array for each family ()i.e., each component of each grid cell, where the ID is the
 *      total number of families before this family while iterating over the grid.
 *  + At each grid cell in each box/tile on each processor:
 *    + Set community number.
 *    + Find unit number for this community; specify that a part of this unit is on this processor;
 *      set unit number, FIPS code, and census tract number at this grid cell (community).
 *    + Set community size: 2000 people, unless this is the last community of a unit, in which case
 *      the remaining people if > 1000 (else 0).
 *    + Compute cumulative distribution (on a scale of 0-1000) of household size ranging from 1 to 7:
 *      initialize with default distributions, then compute from census data if available.
 *    + For each person in this community, generate a random integer between 0 and 1000; based on its
 *      value, assign this person to a household of a certain size (1-7) based on the cumulative
 *      distributions above.
 *  + Compute total number of agents (people), family offsets and IDs over the box/tile.
 *  + Allocate particle container AoS and SoA arrays for the computed number of agents.
 *  + At each grid cell in each box/tile on each processor, and for each component (where component
 *    corresponds to family size):
 *    + Compute percentage of school age kids (kids of age 5-17 as a fraction of total kids - under 5
 *      plus 5-17), if available in census data or set to default (76%).
 *    + For each agent at this grid cell and family size (component):
 *      + Find age group by generating a random integer (0-100) and using default age distributions.
 *        Look at code to see the algorithm for family size > 1.
 *      + Set agent position at the center of this grid cell.
 *      + Initialize status and day counters.
 *      + Set age group and family ID.
 *      + Set home location to current grid cell.
 *      + Initialize work location to current grid cell. Actual work location is set in
 *        ExaEpi::read_workerflow().
 *      + Set neighborhood and work neighborhood values. Actual work neighborhood is set
 *        in ExaEpi::read_workerflow().
 *      + Initialize workgroup to 0. It is set in ExaEpi::read_workerflow().
 *      + If age group is 5-17, assign a school based on neighborhood (#assign_school).
 *  + Copy everything to GPU device.
*/
void AgentContainer::initAgentsCensus (BoxArray &ba, DistributionMapping &dm, DemographicData& demo)
{
    BL_PROFILE("initAgentsCensus");

    ic_type = ExaEpi::ICType::Census;

    const Box& domain = Geom(0).Domain();

    num_residents.define(ba, dm, 6, 0);
    unit_mf.define(ba, dm, 1, 0);
    FIPS_mf.define(ba, dm, 2, 0);
    comm_mf.define(ba, dm, 1, 0);

    num_residents.setVal(0);
    unit_mf.setVal(-1);
    FIPS_mf.setVal(-1);
    comm_mf.setVal(-1);

    iMultiFab num_families(num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    iMultiFab fam_offsets (num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    iMultiFab fam_id (num_residents.boxArray(), num_residents.DistributionMap(), 7, 0);
    num_families.setVal(0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto unit_arr = unit_mf[mfi].array();
        auto FIPS_arr = FIPS_mf[mfi].array();
        auto comm_arr = comm_mf[mfi].array();
        auto nf_arr = num_families[mfi].array();
        auto nr_arr = num_residents[mfi].array();

        auto unit_on_proc = demo.Unit_on_proc_d.data();
        auto Start = demo.Start_d.data();
        auto FIPS = demo.FIPS_d.data();
        auto Tract = demo.Tract_d.data();
        auto Population = demo.Population_d.data();

        auto H1 = demo.H1_d.data();
        auto H2 = demo.H2_d.data();
        auto H3 = demo.H3_d.data();
        auto H4 = demo.H4_d.data();
        auto H5 = demo.H5_d.data();
        auto H6 = demo.H6_d.data();
        auto H7 = demo.H7_d.data();

        auto N5  = demo.N5_d.data();
        auto N17 = demo.N17_d.data();
        //auto N29 = demo.N29_d.data();
        //auto N64 = demo.N64_d.data();
        //auto N65plus = demo.N65plus_d.data();

        auto Ncommunity = demo.Ncommunity;

        auto bx = mfi.tilebox();
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            int community = (int) domain.index(IntVect(AMREX_D_DECL(i, j, k)));
            if (community >= Ncommunity) { return; }
            comm_arr(i, j, k) = community;

            int unit = 0;
            while (community >= Start[unit+1]) { unit++; }
            unit_on_proc[unit] = 1;
            unit_arr(i, j, k) = unit;
            FIPS_arr(i, j, k, 0) = FIPS[unit];
            FIPS_arr(i, j, k, 1) = Tract[unit];

            int community_size;
            if (Population[unit] < (1000 + 2000*(community - Start[unit]))) {
                community_size = 0;  /* Don't set up any residents; workgroup-only */
            }
            else {
                community_size = 2000;   /* Standard 2000-person community */
            }

            int p_hh[7] = {330, 670, 800, 900, 970, 990, 1000};
            int num_hh = H1[unit] + H2[unit] + H3[unit] +
                H4[unit] + H5[unit] + H6[unit] + H7[unit];
            if (num_hh) {
                p_hh[0] = 1000 * H1[unit] / num_hh;
                p_hh[1] = 1000* (H1[unit] + H2[unit]) / num_hh;
                p_hh[2] = 1000* (H1[unit] + H2[unit] + H3[unit]) / num_hh;
                p_hh[3] = 1000* (H1[unit] + H2[unit] + H3[unit] + H4[unit]) / num_hh;
                p_hh[4] = 1000* (H1[unit] + H2[unit] + H3[unit] +
                                 H4[unit] + H5[unit]) / num_hh;
                p_hh[5] = 1000* (H1[unit] + H2[unit] + H3[unit] +
                                 H4[unit] + H5[unit] + H6[unit]) / num_hh;
                p_hh[6] = 1000;
            }

            int npeople = 0;
            while (npeople < community_size + 1) {
                int il  = amrex::Random_int(1000, engine);

                int family_size = 1;
                while (il > p_hh[family_size]) { ++family_size; }
                AMREX_ASSERT(family_size > 0);
                AMREX_ASSERT(family_size <= 7);

                nf_arr(i, j, k, family_size-1) += 1;
                npeople += family_size;
            }

            AMREX_ASSERT(npeople == nf_arr(i, j, k, 0) +
                         2*nf_arr(i, j, k, 1) +
                         3*nf_arr(i, j, k, 2) +
                         4*nf_arr(i, j, k, 3) +
                         5*nf_arr(i, j, k, 4) +
                         6*nf_arr(i, j, k, 5) +
                         7*nf_arr(i, j, k, 6));

            nr_arr(i, j, k, 5) = npeople;
        });

        int nagents;
        int ncomp = num_families[mfi].nComp();
        int ncell = num_families[mfi].numPts();
        {
            BL_PROFILE("setPopulationCounts_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_offsets[mfi].dataPtr();
            nagents = Scan::PrefixSum<int>(ncomp*ncell,
                            [=] AMREX_GPU_DEVICE (int i) -> int {
                                int comp = i / ncell;
                                return (comp+1)*in[i];
                            },
                            [=] AMREX_GPU_DEVICE (int i, int const& x) { out[i] = x; },
                                               Scan::Type::exclusive, Scan::retSum);
        }
        {
            BL_PROFILE("setFamily_id_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_id[mfi].dataPtr();
            Scan::PrefixSum<int>(ncomp*ncell,
                                 [=] AMREX_GPU_DEVICE (int i) -> int {
                                     return in[i];
                                 },
                                 [=] AMREX_GPU_DEVICE (int i, int const& x) { out[i] = x; },
                                 Scan::Type::exclusive, Scan::retSum);
        }

        auto offset_arr = fam_offsets[mfi].array();
        auto fam_id_arr = fam_id[mfi].array();
        auto& agents_tile = GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
        agents_tile.resize(nagents);
        auto aos = &agents_tile.GetArrayOfStructs()[0];
        auto& soa = agents_tile.GetStructOfArrays();

        auto status_ptr = soa.GetIntData(IntIdx::status).data();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto family_ptr = soa.GetIntData(IntIdx::family).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
        auto school_ptr = soa.GetIntData(IntIdx::school).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();

        auto counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();
        auto timer_ptr = soa.GetRealData(RealIdx::treatment_timer).data();
        auto dx = ParticleGeom(0).CellSizeArray();
        auto myproc = MyProc();

        Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (init_agents_nextid)
#endif
        {
            pid = PType::NextID();
            PType::NextID(pid+nagents);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<Long>(pid + nagents) < LastParticleID,
            "Error: overflow on agent id numbers!");

        amrex::ParallelForRNG(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, amrex::RandomEngine const& engine) noexcept
        {
            int nf = nf_arr(i, j, k, n);
            if (nf == 0) return;

            int unit = unit_arr(i, j, k);
            int community = comm_arr(i, j, k);
            int family_id_start = fam_id_arr(i, j, k, n);
            int family_size = n + 1;
            int num_to_add = family_size * nf;

            int community_size;
            if (Population[unit] < (1000 + 2000*(community - Start[unit]))) {
                community_size = 0;  /* Don't set up any residents; workgroup-only */
            }
            else {
                community_size = 2000;   /* Standard 2000-person community */
            }

            int p_schoolage = 0;
            if (community_size) {  // Only bother for residential communities
                if (N5[unit] + N17[unit]) {
                    p_schoolage = 100*N17[unit] / (N5[unit] + N17[unit]);
                }
                else {
                    p_schoolage = 76;
                }
            }

            int start = offset_arr(i, j, k, n);
            int nborhood = 0;
            for (int ii = 0; ii < num_to_add; ++ii) {
                int ip = start + ii;
                auto& agent = aos[ip];
                int il2 = amrex::Random_int(100, engine);
                if (ii % family_size == 0) {
                    nborhood = amrex::Random_int(4, engine);
                }
                int age_group = -1;

                if (family_size == 1) {
                    if (il2 < 28) { age_group = 4; }      /* single adult age 65+   */
                    else if (il2 < 68) { age_group = 3; } /* age 30-64 (ASSUME 40%) */
                    else { age_group = 2; }               /* single adult age 19-29 */
                    nr_arr(i, j, k, age_group) += 1;
                } else if (family_size == 2) {
                    if (il2 == 0) {
                        /* 1% probability of one parent + one child */
                        int il3 = amrex::Random_int(100, engine);
                        if (il3 < 2) { age_group = 4; }        /* one parent, age 65+ */
                        else if (il3 < 62) { age_group = 3; }  /* one parent 30-64 (ASSUME 60%) */
                        else { age_group = 2; }                /* one parent 19-29 */
                        nr_arr(i, j, k, age_group) += 1;
                        if (((int) amrex::Random_int(100, engine)) < p_schoolage) {
                            age_group = 1; /* 22.0% of total population ages 5-18 */
                        } else {
                            age_group = 0;   /* 6.8% of total population ages 0-4 */
                        }
                        nr_arr(i, j, k, age_group) += 1;
                    } else {
                        /* 2 adults, 28% over 65 (ASSUME both same age group) */
                        if (il2 < 28) { age_group = 4; }      /* single adult age 65+ */
                        else if (il2 < 68) { age_group = 3; } /* age 30-64 (ASSUME 40%) */
                        else { age_group = 2; }               /* single adult age 19-29 */
                        nr_arr(i, j, k, age_group) += 2;
                    }
                }

                if (family_size > 2) {
                    /* ASSUME 2 adults, of the same age group */
                    if (il2 < 2) { age_group = 4; }  /* parents are age 65+ */
                    else if (il2 < 62) { age_group = 3; }  /* parents 30-64 (ASSUME 60%) */
                    else { age_group = 2; }  /* parents 19-29 */
                    nr_arr(i, j, k, age_group) += 2;

                    /* Now pick the children's age groups */
                    for (int nc = 2; nc < family_size; ++nc) {
                        if (((int) amrex::Random_int(100, engine)) < p_schoolage) {
                            age_group = 1; /* 22.0% of total population ages 5-18 */
                        } else {
                            age_group = 0;   /* 6.8% of total population ages 0-4 */
                        }
                        nr_arr(i, j, k, age_group) += 1;
                    }
                }

                agent.id()  = pid+ip;
                agent.cpu() = myproc;
                agent.pos(0) = ((Real)i + 0.5_rt) * dx[0];
                agent.pos(1) = ((Real)j + 0.5_rt) * dx[1];

                status_ptr[ip] = 0;
                counter_ptr[ip] = 0.0_rt;
                timer_ptr[ip] = 0.0_rt;
                age_group_ptr[ip] = age_group;
                family_ptr[ip] = family_id_start + (ii / family_size);
                home_i_ptr[ip] = i;
                home_j_ptr[ip] = j;
                work_i_ptr[ip] = i;
                work_j_ptr[ip] = j;
                nborhood_ptr[ip] = nborhood;
                workgroup_ptr[ip] = 0;

                if (age_group == 0) {
                    school_ptr[ip] = 5; // note - need to handle playgroups
                } else if (age_group == 1) {
                    school_ptr[ip] = assign_school(nborhood, engine);
                } else{
                    school_ptr[ip] = -1;
                }
            }
        });
    }

    demo.CopyToHostAsync(demo.Unit_on_proc_d, demo.Unit_on_proc);
    amrex::Gpu::streamSynchronize();
}

/*! \brief Initialize agents for ExaEpi::ICType::UrbanPop
* Each agent belongs to a community, which is a census block group. These are on average 1500 people in size.
* Each community is located at an x,y point in grid space, and an x,y point in the underlying domain.
* In the non-UrbanPop version the communities are distributed uniformly across a 1.0 x 1.0 geometry.
* With the UrbanPop data, we use the latitude and longitude of each block group as the position.
*/
void AgentContainer::initAgentsUrbanPop (UrbanPop::UrbanPopData &urban_pop, const int nborhood_size, const int workgroup_size) {
    BL_PROFILE("initAgentsUrbanPop");

    ic_type = ExaEpi::ICType::UrbanPop;
    min_pos_x = ParticleGeom(0).ProbLo()[0];
    min_pos_y = ParticleGeom(0).ProbLo()[1];

    // only a single level
    auto& particles  = GetParticles(0);
    // collect all the block groups per box
    std::unordered_map<int, Vector<UrbanPop::BlockGroup>> box_block_groups;
    for (auto block_group : urban_pop.block_groups) {
        box_block_groups[block_group.box_i].push_back(block_group);
    }
    int tot_np = 0;
    // don't tile here because the UrbanPop data is stored in a non-tiled, per box basis.
    for (MFIter mfi = MakeMFIter(0, false); mfi.isValid(); ++mfi) {
        int box_i = mfi.index();
        int tile_i = mfi.LocalTileIndex();
        auto bx = mfi.tilebox();
        auto &ptile = particles[std::make_pair(box_i, tile_i)];
        auto &block_groups = box_block_groups[box_i];
        if (block_groups.empty()) continue;
        int tot_pop = 0;
        for (auto &block_group : block_groups) {
            tot_pop += block_group.people.size();
        }
        ptile.resize(tot_pop);
        auto dx = ParticleGeom(0).CellSizeArray();
        auto aos = &ptile.GetArrayOfStructs()[0];
        auto &soa = ptile.GetStructOfArrays();
        auto status_ptr = soa.GetIntData(IntIdx::status).data();
        auto counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();
        auto timer_ptr = soa.GetRealData(RealIdx::treatment_timer).data();
        auto family_ptr = soa.GetIntData(IntIdx::family).data();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
        auto school_ptr = soa.GetIntData(IntIdx::school).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto fips_ptr = soa.GetIntData(IntIdx::fips).data();

        int my_proc = MyProc();
        int block_pi = 0;
        for (auto &block_group : block_groups) {
            tot_np += block_group.people.size();
            int x = block_group.x;
            int y = block_group.y;
            Real px = (Real)x * dx[0] + min_pos_x;
            Real py = (Real)y * dx[1] + min_pos_y;
            int n = block_group.people.size();
            // set number of nbhoods to get each nbhood as close to nborhood_size as possible
            int num_nbhoods = std::max(std::round((double)n / nborhood_size), 1.0);
            auto people = &block_group.people[0];
            RandomEngine engine;
            // randomly shuffle work locations in this block group (Fisher-Yates shuffle)
            // FIXME: this should be done in GPU, but not sure how to synchronize at AMREX level
            //ParallelForRNG(block_group.people.size(), [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
            for (int i = n - 1; i > 0; i--) {
                auto &person = people[i];
                if (!person.is_worker()) continue;
                int target = Random_int(i, engine);
                auto &target_person = people[target];
                if (!target_person.is_worker()) continue;
                std::swap(person.work_x, target_person.work_x);
                std::swap(person.work_y, target_person.work_y);
                std::swap(person.w_geoid, target_person.w_geoid);
            }
            ParallelForRNG(n, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                auto &person = people[i];
                int pi = block_pi + i;
                auto &agent = aos[pi];
                agent.id()  = person.p_id;
                agent.cpu() = my_proc;
                agent.pos(0) = px;
                agent.pos(1) = py;

                status_ptr[pi] = 0;
                counter_ptr[pi] = 0.0_rt;
                timer_ptr[pi] = 0.0_rt;

                auto age = person.pr_age;
                // Age group (under 5, 5-17, 18-29, 30-64, 65+)
                if (age < 5) age_group_ptr[pi] = 0;
                else if (age < 17) age_group_ptr[pi] = 1;
                else if (age < 29) age_group_ptr[pi] = 2;
                else if (age < 64) age_group_ptr[pi] = 3;
                else age_group_ptr[pi] = 4;

                family_ptr[pi] = person.h_id;
                home_i_ptr[pi] = x;
                home_j_ptr[pi] = y;
                // choose new nbhood for next household
                int nborhood = -1;
                // set this randomly only for the first position in the family - a second loop through will set the other values
                // this approach is needed for GPU parallelism
                if (i == 0 || (people[i].h_id != people[i - 1].h_id)) nborhood = Random_int(num_nbhoods, engine);
                nborhood_ptr[pi] = nborhood;
                if (people[i].pr_emp_stat == 2 || people[i].pr_emp_stat == 3) {
                    work_i_ptr[pi] = people[i].work_x;
                    work_j_ptr[pi] = people[i].work_y;
                } else {
                    work_i_ptr[pi] = home_i_ptr[pi];
                    work_j_ptr[pi] = home_j_ptr[pi];
                    // indicates that the agent doesn't work
                    workgroup_ptr[pi] = 0;
                }

                if (age_group_ptr[pi] == 0) school_ptr[pi] = 5; // note - need to handle playgroups
                else if (age_group_ptr[pi] == 1) school_ptr[pi] = assign_school(nborhood, engine);
                else school_ptr[pi] = -1;
                // FIPS is the first 5 digits of the GEOID, which is 12 digits
                fips_ptr[pi] = (int)(people[i].h_geoid / 10000000);
            });
            // separate loop for setting the workgroup and randomizing workgroups
            for (int i = 0; i < n; i++) {
                auto &person = block_group.people[i];
                int pi = block_pi + i;
                if (person.is_worker()) {
                    int num_workgroups = max(round(urban_pop.block_group_workers[person.w_geoid] / workgroup_size), 1.0);
                    // workgroups are at least 1 to indicate worker
                    workgroup_ptr[pi] = Random_int(num_workgroups, engine) + 1;
                }
            }
            // loop to set the household neighborhoods
            ParallelFor(n, [=] AMREX_GPU_DEVICE (int i) noexcept {
                int pi = block_pi + i;
                if (nborhood_ptr[pi] == -1) {
                    for (int j = pi - 1; j >= 0; j--) {
                        if (nborhood_ptr[j] != -1) {
                            nborhood_ptr[pi] = nborhood_ptr[j];
                            break;
                        }
                    }
                }
            });
            block_pi += n;
        }
        //AllPrint() << MyProc() << ": box " << bx << " box_i " << box_i << " population " << ptile.size() << "\n";
    }
    AllPrint() << "Process " << MyProc() << " has " << particles.size() << " boxes with a total of "
               << tot_np << " particles for " << urban_pop.block_groups.size() << " block groups\n";
}

/*! \brief Send agents on a random walk around the neighborhood

    For each agent, set its position to a random one near its current position
*/
/*void AgentContainer::moveAgentsRandomWalk ()
{
    BL_PROFILE("AgentContainer::moveAgentsRandomWalk");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                ParticleType& p = pstruct[i];
                p.pos(0) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[0]);
                p.pos(1) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[1]);
            });
        }
    }
}*/

/*! \brief Move agents to work

    For each agent, set its position to the work community (IntIdx::work_i, IntIdx::work_j)
*/
void AgentContainer::moveAgentsToWork ()
{
    BL_PROFILE("AgentContainer::moveAgentsToWork");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();

            short _ic_type = ic_type;
            Real _min_pos_x = min_pos_x, _min_pos_y = min_pos_y;

            amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                ParticleType& p = pstruct[ip];
                set_particle_pos(p.pos(0), p.pos(1), work_i_ptr[ip], work_j_ptr[ip], dx[0], dx[1], _ic_type, _min_pos_x, _min_pos_y);
            });
        }
    }

    m_at_work = true;
}

/*! \brief Move agents to home

    For each agent, set its position to the home community (IntIdx::home_i, IntIdx::home_j)
*/
void AgentContainer::moveAgentsToHome ()
{
    BL_PROFILE("AgentContainer::moveAgentsToHome");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            short _ic_type = ic_type;
            Real _min_pos_x = min_pos_x, _min_pos_y = min_pos_y;

            amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                ParticleType& p = pstruct[ip];
                set_particle_pos(p.pos(0), p.pos(1), home_i_ptr[ip], home_j_ptr[ip], dx[0], dx[1], _ic_type, _min_pos_x, _min_pos_y);
            });
        }
    }

    m_at_work = false;
}

/*! \brief Move agents randomly

    For each agent, set its position to a random location with a probabilty of 0.01%
*/
void AgentContainer::moveRandomTravel ()
{
    BL_PROFILE("AgentContainer::moveRandomTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                ParticleType& p = pstruct[i];

                if (amrex::Random(engine) < 0.0001) {
                    p.pos(0) = 3000*amrex::Random(engine);
                    p.pos(1) = 3000*amrex::Random(engine);
                }
            });
        }
    }
}

/*! \brief Updates disease status of each agent at a given step and also updates a MultiFab
    that tracks disease statistics (hospitalization, ICU, ventilator, and death) in a community.

    At a given step, update the disease status of each agent based on the following overall logic:
    + If agent status is #Status::never or #Status::susceptible, do nothing
    + If agent status is #Status::infected, then
      + Increment its counter by 1 day
      + If counter is within incubation period (#DiseaseParm::incubation_length days), do nothing more
      + Else on day #DiseaseParm::incubation_length, use hospitalization probabilities (by age group)
        to decide if agent is hospitalized. If yes, use age group to set hospital timer. Also, use
        age-group-wise probabilities to move agent to ICU and then to ventilator. Adjust timer
        accordingly.
      + Update the community-wise disease stats tracker MultiFab according to hospitalization/ICU/vent
        status (using the agent's home community)
      + Else (beyond 3 days), count down hospital timer if agent is hospitalized. At end of hospital
        stay, determine if agent is #Status dead or #Status::immune. For non-hospitalized agents,
        set them to #Status::immune after #DiseaseParm::incubation_length +
        #DiseaseParm::infectious_length days.

    The input argument is a MultiFab with 4 components corresponding to "hospitalizations", "ICU",
    "ventilator", and "death". It contains the cumulative totals of these quantities for each
    community as the simulation progresses.
*/
void AgentContainer::updateStatus (MultiFab& disease_stats /*!< Community-wise disease stats tracker */)
{
    BL_PROFILE("AgentContainer::updateStatus");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();
            auto timer_ptr = soa.GetRealData(RealIdx::treatment_timer).data();
            auto prob_ptr = soa.GetRealData(RealIdx::prob).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto symptomatic_ptr = soa.GetIntData(IntIdx::symptomatic).data();
            auto incubation_period_ptr = soa.GetRealData(RealIdx::incubation_period).data();
            auto infectious_period_ptr = soa.GetRealData(RealIdx::infectious_period).data();
            auto symptomdev_period_ptr = soa.GetRealData(RealIdx::symptomdev_period).data();

            auto* lparm = d_parm;

            auto ds_arr = disease_stats[mfi].array();

            struct DiseaseStats
            {
                enum {
                    hospitalization = 0,
                    ICU,
                    ventilator,
                    death
                };
            };

            auto symptomatic_withdraw = m_symptomatic_withdraw;
            auto symptomatic_withdraw_compliance = m_symptomatic_withdraw_compliance;

            auto mean_immune_time = h_parm->mean_immune_time;
            auto immune_time_spread = h_parm->immune_time_spread;

            // Track hospitalization, ICU, ventilator, and fatalities
            Real CHR[] = {.0104_rt, .0104_rt, .070_rt, .28_rt, 1.0_rt};  // sick -> hospital probabilities
            Real CIC[] = {.24_rt, .24_rt, .24_rt, .36_rt, .35_rt};      // hospital -> ICU probabilities
            Real CVE[] = {.12_rt, .12_rt, .12_rt, .22_rt, .22_rt};      // ICU -> ventilator probabilities
            Real CVF[] = {.20_rt, .20_rt, .20_rt, 0.45_rt, 1.26_rt};    // ventilator -> dead probilities
            amrex::ParallelForRNG( np,
                                   [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
                prob_ptr[i] = 1.0_rt;
                if ( status_ptr[i] == Status::never ||
                     status_ptr[i] == Status::susceptible ) {
                    return;
                }
                else if (status_ptr[i] == Status::immune) {
                    counter_ptr[i] -= 1.0_rt;
                    if (counter_ptr[i] < 0.0_rt) {
                        counter_ptr[i] = 0.0_rt;
                        timer_ptr[i] = 0.0_rt;
                        status_ptr[i] = Status::susceptible;
                        return;
                    }
                }
                else if (status_ptr[i] == Status::infected) {
                    counter_ptr[i] += 1;
                    if (counter_ptr[i] == 1) {
                        if (amrex::Random(engine) < lparm->p_asymp[0]) {
                            symptomatic_ptr[i] = SymptomStatus::asymptomatic;
                        } else {
                            symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                        }
                    }
                    if (counter_ptr[i] == amrex::Math::floor(symptomdev_period_ptr[i])) {
                        if (symptomatic_ptr[i] != SymptomStatus::asymptomatic) {
                            symptomatic_ptr[i] = SymptomStatus::symptomatic;
                        }
                        if (    (symptomatic_ptr[i] == SymptomStatus::symptomatic)
                            &&  (symptomatic_withdraw)
                            &&  (amrex::Random(engine) < symptomatic_withdraw_compliance)) {
                            withdrawn_ptr[i] = 1;
                        }
                    }
                    if (counter_ptr[i] < incubation_period_ptr[i]) {
                        // incubation phase
                        return;
                    }
                    if (counter_ptr[i] == amrex::Math::ceil(incubation_period_ptr[i])) {
                        // decide if hospitalized
                        Real p_hosp = CHR[age_group_ptr[i]];
                        if (amrex::Random(engine) < p_hosp) {
                            if ((age_group_ptr[i]) < 3) {  // age groups 0-4, 5-18, 19-29
                                timer_ptr[i] = 3;  // Ages 0-49 hospitalized for 3.1 days
                            }
                            else if (age_group_ptr[i] == 4) {
                                timer_ptr[i] = 7;  // Age 65+ hospitalized for 6.5 days
                            }
                            else if (amrex::Random(engine) < 0.57) {
                                timer_ptr[i] = 3;  // Proportion of 30-64 that is under 50
                            }
                            else {
                                timer_ptr[i] = 8;  // Age 50-64 hospitalized for 7.8 days
                            }
                            amrex::Gpu::Atomic::AddNoRet(
                                &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                        DiseaseStats::hospitalization), 1.0_rt);
                            if (amrex::Random(engine) < CIC[age_group_ptr[i]]) {
                                //std::printf("putting h in icu \n");
                                timer_ptr[i] += 10;  // move to ICU
                                amrex::Gpu::Atomic::AddNoRet(
                                    &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                            DiseaseStats::ICU), 1.0_rt);
                                if (amrex::Random(engine) < CVE[age_group_ptr[i]]) {
                                    //std::printf("putting icu on v \n");
                                    amrex::Gpu::Atomic::AddNoRet(
                                    &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                            DiseaseStats::ventilator), 1.0_rt);
                                    timer_ptr[i] += 10;  // put on ventilator
                                }
                            }
                        }
                    } else {
                        if (timer_ptr[i] > 0.0_rt) {
                            // do hospital things
                            timer_ptr[i] -= 1.0_rt;
                            if (timer_ptr[i] == 0) {
                                if (CVF[age_group_ptr[i]] > 2.0_rt) {
                                    if (amrex::Random(engine) < (CVF[age_group_ptr[i]] - 2.0_rt)) {
                                        amrex::Gpu::Atomic::AddNoRet(
                                            &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                    DiseaseStats::death), 1.0_rt);
                                        status_ptr[i] = Status::dead;
                                    }
                                }
                                amrex::Gpu::Atomic::AddNoRet(
                                                             &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                                     DiseaseStats::hospitalization), -1.0_rt);
                                if (status_ptr[i] != Status::dead) {
                                    status_ptr[i] = Status::immune;  // If alive, hospitalized patient recovers
                                    counter_ptr[i] = (mean_immune_time - immune_time_spread) + 2.0_rt*immune_time_spread*amrex::Random(engine);
                                    symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                    withdrawn_ptr[i] = 0;
                                }
                            }
                            if (timer_ptr[i] == 10) {
                                if (CVF[age_group_ptr[i]] > 1.0_rt) {
                                    if (amrex::Random(engine) < (CVF[age_group_ptr[i]] - 1.0_rt)) {
                                        amrex::Gpu::Atomic::AddNoRet(
                                            &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                    DiseaseStats::death), 1.0_rt);
                                        status_ptr[i] = Status::dead;
                                    }
                                }
                                amrex::Gpu::Atomic::AddNoRet(
                                                             &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                                     DiseaseStats::hospitalization), -1.0_rt);
                                amrex::Gpu::Atomic::AddNoRet(
                                                             &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                                     DiseaseStats::ICU), -1.0_rt);
                                if (status_ptr[i] != Status::dead) {
                                    status_ptr[i] = Status::immune;  // If alive, ICU patient recovers
                                    counter_ptr[i] = (mean_immune_time - immune_time_spread) + 2.0_rt*immune_time_spread*amrex::Random(engine);
                                    symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                    withdrawn_ptr[i] = 0;
                                }
                            }
                            if (timer_ptr[i] == 20) {
                                if (amrex::Random(engine) < CVF[age_group_ptr[i]]) {
                                    amrex::Gpu::Atomic::AddNoRet(
                                        &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                DiseaseStats::death), 1.0_rt);
                                    status_ptr[i] = Status::dead;
                                }
                                amrex::Gpu::Atomic::AddNoRet(
                                                             &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                                     DiseaseStats::hospitalization), -1.0_rt);
                                amrex::Gpu::Atomic::AddNoRet(
                                                             &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                                     DiseaseStats::ICU), -1.0_rt);
                                amrex::Gpu::Atomic::AddNoRet(
                                                             &ds_arr(home_i_ptr[i], home_j_ptr[i], 0,
                                                                     DiseaseStats::ventilator), -1.0_rt);
                                if (status_ptr[i] != Status::dead) {
                                    status_ptr[i] = Status::immune;  // If alive, ventilated patient recovers
                                counter_ptr[i] = (mean_immune_time - immune_time_spread) + 2.0_rt*immune_time_spread*amrex::Random(engine);
                                symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                withdrawn_ptr[i] = 0;
                                }
                            }
                        }
                        else { // not hospitalized, recover once not infectious
                            if (counter_ptr[i] >= (incubation_period_ptr[i] + infectious_period_ptr[i])) {
                                status_ptr[i] = Status::immune;
                                counter_ptr[i] = (mean_immune_time - immune_time_spread) + 2.0_rt*immune_time_spread*amrex::Random(engine);
                                symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                withdrawn_ptr[i] = 0;
                            }
                        }
                    }
                }
            });
        }
    }
}

/*! \brief Start shelter-in-place */
void AgentContainer::shelterStart ()
{
    BL_PROFILE("AgentContainer::shelterStart");

    Print() << "Starting shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            auto shelter_compliance = m_shelter_compliance;
            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
                if (amrex::Random(engine) < shelter_compliance) {
                    withdrawn_ptr[i] = 1;
                }
            });
        }
    }
}

/*! \brief Stop shelter-in-place */
void AgentContainer::shelterStop ()
{
    BL_PROFILE("AgentContainer::shelterStop");

    Print() << "Stopping shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                withdrawn_ptr[i] = 0;
            });
        }
    }
}

/*! \brief Infect agents based on their current status and the computed probability of infection.
    The infection probability is computed in AgentContainer::interactAgentsHomeWork() or
    AgentContainer::interactAgents() */
void AgentContainer::infectAgents ()
{
    BL_PROFILE("AgentContainer::infectAgents");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();
            auto prob_ptr = soa.GetRealData(RealIdx::prob).data();
            auto incubation_period_ptr = soa.GetRealData(RealIdx::incubation_period).data();
            auto infectious_period_ptr = soa.GetRealData(RealIdx::infectious_period).data();
            auto symptomdev_period_ptr = soa.GetRealData(RealIdx::symptomdev_period).data();

            auto* lparm = d_parm;

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
                prob_ptr[i] = 1.0_rt - prob_ptr[i];
                if ( status_ptr[i] == Status::never ||
                     status_ptr[i] == Status::susceptible ) {
                    if (amrex::Random(engine) < prob_ptr[i]) {
                        status_ptr[i] = Status::infected;
                        counter_ptr[i] = 0.0_rt;
                        incubation_period_ptr[i] = amrex::RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                        infectious_period_ptr[i] = amrex::RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                        symptomdev_period_ptr[i] = amrex::RandomNormal(lparm->symptomdev_length_mean, lparm->symptomdev_length_std, engine);
                        return;
                    }
                }
            });
        }
    }
}

/*! \brief Computes the number of agents with various #Status in each grid cell of the
    computational domain.

    Given a MultiFab with at least 5 components that is defined with the same box array and
    distribution mapping as this #AgentContainer, the MultiFab will contain (at the end of
    this function) the following *in each cell*:
    + component 0: total number of agents in this grid cell.
    + component 1: number of agents that have never been infected (#Status::never)
    + component 2: number of agents that are infected (#Status::infected)
    + component 3: number of agents that are immune (#Status::immune)
    + component 4: number of agents that are susceptible infected (#Status::susceptible)
*/
void AgentContainer::generateCellData (MultiFab& mf /*!< MultiFab with at least 5 components */) const
{
    BL_PROFILE("AgentContainer::generateCellData");

    const int lev = 0;

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();
    amrex::ParticleToMesh(*this, mf, lev,
        [=] AMREX_GPU_DEVICE (const SuperParticleType& p,
                              amrex::Array4<amrex::Real> const& count)
        {
            int status = p.idata(0);
            auto iv = getParticleCell(p, plo, dxi, domain);
            amrex::Gpu::Atomic::AddNoRet(&count(iv, 0), 1.0_rt);
            if (status == Status::never) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 1), 1.0_rt);
            }
            else if (status == Status::infected) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 2), 1.0_rt);
            }
            else if (status == Status::immune) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 3), 1.0_rt);
            }
            else if (status == Status::susceptible) {
                amrex::Gpu::Atomic::AddNoRet(&count(iv, 4), 1.0_rt);
            }
        }, false);
}

/*! \brief Computes the total number of agents with each #Status

    Returns a vector with 5 components corresponding to each value of #Status; each element is
    the total number of agents at a step with the corresponding #Status (in that order).
*/
std::array<Long, 9> AgentContainer::getTotals () {
    BL_PROFILE("AgentContainer::getTotals");
    amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<int,int,int,int,int,int,int,int,int>> (
                  *this, [=] AMREX_GPU_DEVICE (const SuperParticleType& p) noexcept
                  -> amrex::GpuTuple<int,int,int,int,int,int,int,int,int>
              {
                  int s[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                  AMREX_ALWAYS_ASSERT(p.idata(IntIdx::status) >= 0);
                  AMREX_ALWAYS_ASSERT(p.idata(IntIdx::status) <= 4);
                  s[p.idata(IntIdx::status)] = 1;
                  if (p.idata(IntIdx::status) == 1) {  // exposed
                      if (p.rdata(RealIdx::disease_counter) <= p.rdata(RealIdx::incubation_period)) {
                          s[5] = 1;  // exposed, but not infectious
                      } else { // infectious
                          if (p.idata(IntIdx::symptomatic) == SymptomStatus::asymptomatic) {
                              s[6] = 1;  // asymptomatic and will remain so
                          }
                          else if (p.idata(IntIdx::symptomatic) == SymptomStatus::presymptomatic) {
                              s[7] = 1;  // asymptomatic but will develop symptoms
                          }
                          else if (p.idata(IntIdx::symptomatic) == SymptomStatus::symptomatic) {
                              s[8] = 1;  // Infectious and symptomatic
                          } else {
                              amrex::Abort("how did I get here?");
                          }
                      }
                  }
                  return {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]};
              }, reduce_ops);

    std::array<Long, 9> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r),
                                  amrex::get<4>(r), amrex::get<5>(r), amrex::get<6>(r), amrex::get<7>(r),
                                  amrex::get<8>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 9, ParallelDescriptor::IOProcessorNumber());
    return counts;
}

/*! \brief Interaction and movement of agents during morning commute
 *
 * + Move agents to work
 * + Simulate interactions during morning commute (public transit/carpool/etc ?)
*/
void AgentContainer::morningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::morningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToWork();
}

/*! \brief Interaction and movement of agents during evening commute
 *
 * + Simulate interactions during evening commute (public transit/carpool/etc ?)
 * + Simulate interactions at locations agents may stop by on their way home
 * + Move agents to home
*/
void AgentContainer::eveningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::eveningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    //if (haveInteractionModel(ExaEpi::InteractionNames::grocery_store)) {
    //    m_interactions[ExaEpi::InteractionNames::grocery_store]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToHome();
}

/*! \brief Interaction of agents during day time - work and school */
void AgentContainer::interactDay ( MultiFab& a_mask_behavior /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactDay");
    if (haveInteractionModel(ExaEpi::InteractionNames::work)) {
        m_interactions[ExaEpi::InteractionNames::work]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::school)) {
        m_interactions[ExaEpi::InteractionNames::school]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::nborhood)) {
        m_interactions[ExaEpi::InteractionNames::nborhood]->interactAgents( *this, a_mask_behavior );
    }
}

/*! \brief Interaction of agents during evening (after work) - social stuff */
void AgentContainer::interactEvening ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactEvening");
}

/*! \brief Interaction of agents during nighttime time - at home */
void AgentContainer::interactNight ( MultiFab& a_mask_behavior /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::interactNight");
    if (haveInteractionModel(ExaEpi::InteractionNames::home)) {
        m_interactions[ExaEpi::InteractionNames::home]->interactAgents( *this, a_mask_behavior );
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::nborhood)) {
        m_interactions[ExaEpi::InteractionNames::nborhood]->interactAgents( *this, a_mask_behavior );
    }
}


void AgentContainer::writeAgentsFile (const string &fname, int step_number) {
    BL_PROFILE("AgentContainer::writeAgentsFile");
    string my_fname = fname;
    if (step_number != -1) my_fname += ".s" + std::to_string(step_number);
    my_fname += "." + std::to_string(MyProc()) + ".tsv";
    Print() << "Writing agents to files " << my_fname + "\n";
    std::ofstream outfs(my_fname);
    if (step_number == -1)
        outfs << "#ID\tx-position\ty-position\tfamily\tage\thome\twork\tnbh\tschl\tworkg\tfips\n";
    else
        outfs << "#ID\tx-position\ty-position\thome\tstatus\n";
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev  = GetParticles(lev);
        int max_x = 0, max_y = 0;
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa = ptile.GetStructOfArrays();
            auto& aos = ptile.GetArrayOfStructs();
            const auto np = ptile.numParticles();
            auto family_ptr = soa.GetIntData(IntIdx::family).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
            auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
            auto school_ptr = soa.GetIntData(IntIdx::school).data();
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
            auto fips_ptr = soa.GetIntData(IntIdx::fips).data();
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            for (int i = 0; i < np; i++) {
                auto& agent = aos[i];
                if (step_number != -1) {
                    outfs << agent.id() << "\t" << std::fixed << std::setprecision(8)
                          << agent.pos(0) << "\t" << agent.pos(1) << "\t"
                          << home_i_ptr[i] << "," << home_j_ptr[i] << "\t" << status_ptr[i] << "\n";
                } else {
                    outfs << agent.id() << "\t" << std::fixed << std::setprecision(8)
                          << agent.pos(0) << "\t" << agent.pos(1) << "\t"
                          << family_ptr[i] << "\t" << age_group_ptr[i] << "\t"
                          << home_i_ptr[i] << "," << home_j_ptr[i] << "\t"
                          << work_i_ptr[i] << "," << work_j_ptr[i] << "\t"
                          << nborhood_ptr[i] << "\t" << school_ptr[i] << "\t" << workgroup_ptr[i] << "\t"
                          << fips_ptr[i] << "\n";
                }
                max_x = std::max(max_x, home_i_ptr[i]);
                max_y = std::max(max_y, home_j_ptr[i]);
            }
        }
#ifdef DEBUG
        if (step_number == -1) {
            Vector<Vector<int>> communities(max_x + 1, Vector<int>(max_y + 1, 0));
            int tot_np = 0;
            for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                //AllPrint() << MyProc() << ": gid " << gid << " tid " << tid << "\n";
                auto& ptile = plev[std::make_pair(gid, tid)];
                auto& soa = ptile.GetStructOfArrays();
                auto& aos = ptile.GetArrayOfStructs();
                const auto np = ptile.numParticles();
                auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
                auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
                for (int i = 0; i < np; i++) {
                    auto& agent = aos[i];
                    communities[home_i_ptr[i]][home_j_ptr[i]]++;
                }
                tot_np += np;
            }
            string my_fname_communities = fname + ".communities." + std::to_string(MyProc()) + ".tsv";
            std::ofstream outfs_communities(my_fname_communities);
            for (int x = 0; x <= max_x; x++) {
                for (int y = 0; y <= max_y; y++) {
                    if (communities[x][y] > 0) outfs_communities << x << "\t" << y << "\t" << communities[x][y] << "\n";
                }
            }
            outfs_communities.close();
            AllPrint() << "tot np " << tot_np << "\n";
        }
#endif
    }
    outfs.close();
}

void AgentContainer::infectAgents (const CaseData &cases) {
    BL_PROFILE("AgentContainer::infectAgents");
    /*
    // get the total number of candidates to be infected
    amrex::ReduceOps<ReduceOpSum> reduce_ops;
    auto r = ParticleReduce<ReduceData<int>>(*this, [=] AMREX_GPU_DEVICE (const SuperParticleType& p) noexcept -> GpuTuple<int> {
        int s = 0;
        if (cases.num_cases[p.idata(IntIdx::fips)]) s = 1;
        return {s};
    }, reduce_ops);
    int counts = amrex::get<0>(r);
    ParallelDescriptor::ReduceIntSum(counts);
    AllPrint() << "reduction count " << counts << "\n";
    */
    // infect with given probability
    // FIXME: this isn't accurate enough. Need to actually randomly pick agents until we have enough
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev  = GetParticles(lev);
/*#ifdef AMREX_USE_GPU
        int *num_candidates = (int*)amrex::The_Arena()->alloc(cases.num_cases.size() * sizeof(int));
        int *num_cases = (int*)amrex::The_Arena()->alloc(cases.num_cases.size() * sizeof(int));
        Gpu::htod_memcpy(num_cases, cases.num_cases.data(), cases.num_cases.size() * sizeof(int));
#else*/
        int *num_candidates = new int[cases.num_cases.size()];
        const int *num_cases = cases.num_cases.data();
//#endif

        // first count up the number of candidates in each FIPS code, for each process
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto fips_ptr = soa.GetIntData(IntIdx::fips).data();
            for (int i = 0; i < np; i++) {
                if (cases.num_cases[fips_ptr[i]]) num_candidates[fips_ptr[i]]++;
            }
        }
        // sum up the number of candidates for each FIPS code
        // FIXME: this is one reduction per FIPS code, which may be really excessive if we have a wide distribution of initial
        // infections
        int tot_num_infections = 0;
        for (int i = 0; i < cases.num_cases.size(); i++) {
            if (cases.num_cases[i]) {
                int tot_num_candidates = num_candidates[i];
                ParallelDescriptor::ReduceIntSum(tot_num_candidates);
                AllPrint() << "Process " << MyProc() << " FIPS " << i << " cases " << cases.num_cases[i]
                           << " candidates " << num_candidates[i] << " total " << tot_num_candidates << "\n";
                num_candidates[i] = tot_num_candidates;
                tot_num_infections += cases.num_cases[i];
            }
        }
        const auto* lparm = getDiseaseParameters_d();
        // infect with probability determined by the total number of candidates and infections
        int num_infected = 0;
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto fips_ptr = soa.GetIntData(IntIdx::fips).data();
            auto status_ptr = soa.GetIntData(IntIdx::status).data();
            auto counter_ptr = soa.GetRealData(RealIdx::disease_counter).data();
            auto incubation_period_ptr = soa.GetRealData(RealIdx::incubation_period).data();
            auto infectious_period_ptr = soa.GetRealData(RealIdx::infectious_period).data();
            auto symptomdev_period_ptr = soa.GetRealData(RealIdx::symptomdev_period).data();

            //Gpu::DeviceScalar<int> num_infected_d(num_infected);
            //int* num_infected_p = num_infected_d.dataPtr();

            //ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
            RandomEngine engine;
            for (int i = 0; i < np; i++) {
                auto cases_in_fips = num_cases[fips_ptr[i]];
                if (cases_in_fips) {
                    if (Random_int(num_candidates[fips_ptr[i]], engine) < cases_in_fips) {
                        status_ptr[i] = Status::infected;
                        counter_ptr[i] = 0;
                        incubation_period_ptr[i] = RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                        infectious_period_ptr[i] = RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                        symptomdev_period_ptr[i] = RandomNormal(lparm->symptomdev_length_mean, lparm->symptomdev_length_std, engine);
                        //*num_infected_p = 1;
                        num_infected++;
                    }
                }
            }//);
            //Gpu::Device::streamSynchronize();
            //num_infected += num_infected_d.dataValue();
        }
        AllPrint() << "Process " << MyProc() << " number infected " << num_infected << "\n";
        ParallelDescriptor::ReduceIntSum(num_infected);
        Print() << "Actual total number infected " << num_infected << " instead of " << tot_num_infections << " cases\n";
    }

}
