/*! @file Utils.cpp
    \brief Contains function implementations for the #ExaEpi::Utils namespace
*/

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_CoordSys.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>


#include "DemographicData.H"
#include "Utils.H"

#include <cmath>
#include <string>

using namespace amrex;
using namespace ExaEpi;

/*! \brief Read in test parameters in #ExaEpi::TestParams from input file */
void ExaEpi::Utils::get_test_params (   TestParams& params,         /*!< Test parameters */
                                        const std::string& prefix   /*!< ParmParse prefix */ )
{
    ParmParse pp(prefix);
    params.size = {1, 1};
    pp.query("size", params.size);

    params.max_grid_size = 16;
    pp.query("max_grid_size", params.max_grid_size);

    pp.get("nsteps", params.nsteps);

    params.plot_int = -1;
    pp.query("plot_int", params.plot_int);

    params.random_travel_int = -1;
    pp.query("random_travel_int", params.random_travel_int);

    std::string ic_type = "demo";
    pp.query( "ic_type", ic_type );
    if (ic_type == "demo") {
        params.ic_type = ICType::Demo;
    } else if (ic_type == "census") {
        params.ic_type = ICType::Census;
        pp.get("census_filename", params.census_filename);
        pp.get("workerflow_filename", params.workerflow_filename);
        pp.get("initial_case_type", params.initial_case_type);
        if (params.initial_case_type == "file") {
            pp.get("case_filename", params.case_filename);
        } else if (params.initial_case_type == "random") {
            pp.get("num_initial_cases", params.num_initial_cases);
        } else {
            amrex::Abort("initial case type not recognized");
        }
    } else if (ic_type == "urbanpop") {
        params.ic_type = ICType::UrbanPop;
        pp.get("urbanpop_filename", params.urbanpop_filename);
    } else {
        amrex::Abort("ic type not recognized");
    }

    params.aggregated_diag_int = -1;
    pp.query("aggregated_diag_int", params.aggregated_diag_int);
    if (params.aggregated_diag_int >= 0) {
        pp.get("aggregated_diag_prefix", params.aggregated_diag_prefix);
    }

    pp.query("shelter_start",  params.shelter_start);
    pp.query("shelter_length", params.shelter_length);

    Long seed = 0;
    bool reset_seed = pp.query("seed", seed);
    if (reset_seed) {
        ULong gpu_seed = (ULong) seed;
        ULong cpu_seed = (ULong) seed;
        amrex::ResetRandomSeed(cpu_seed, gpu_seed);
    }
}

/*! \brief Set computational domain, i.e., number of cells in each direction, from the
    demographic data (number of communities).
 *
 *  If the initialization type (ExaEpi::TestParams::ic_type) is ExaEpi::ICType::Census or ExaEpi::ICType::UrbanPop, then
 *  + The domain is a 2D square, where the total number of cells is the lowest square of an
 *    integer that is greater than #DemographicData::Ncommunity or #UrbanPopData::all_num_block_groups
 *  + The physical size is 1.0 in each dimension.
 *
 *  A periodic Cartesian grid is defined.
*/
Geometry ExaEpi::Utils::get_geometry (int Ncommunities,      /*!< from demographic or UrbanPop data */
                                      const TestParams& params  /*!< test parameters */ ) {
    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++) {
        is_per[i] = true;
    }

    RealBox real_box_latlong;
    // latitude
    real_box_latlong.setLo(0, 31.792004);
    real_box_latlong.setHi(0, 36.959385);
    // longitude
    real_box_latlong.setLo(1, -109.00478);
    real_box_latlong.setHi(1, -103.053505);

    IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect dom_hi(AMREX_D_DECL(100, 100, 100));
    Box base_domain_latlong(dom_lo, dom_hi);
    Geometry geom_latlong;
    int non_periodic[] = {false, false};
    geom_latlong.define(base_domain_latlong, &real_box_latlong, CoordSys::cartesian, non_periodic);

    amrex::Print() << "latlong base domain: " << geom_latlong.Domain()
                   << " cell size array " << geom_latlong.CellSizeArray()[0] << ", "  << geom_latlong.CellSizeArray()[1] << "\n";

    /*BoxArray ba;
    DistributionMapping dm;
    ba.define(geom.Domain());
    ba.maxSize(params.max_grid_size);
    dm.define(ba);

    // each grid point in a box corresponds to a community
    amrex::Print() << "Base domain is: " << geom.Domain() << "\n";
    amrex::Print() << "Max grid size is: " << params.max_grid_size << "\n";
    amrex::Print() << "Number of boxes is: " << ba.size() << " over " << ParallelDescriptor::NProcs() << " ranks. \n";
    */

    RealBox real_box;
    Box base_domain;
    Geometry geom;

    if (params.ic_type == ICType::Demo) {
        IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
        IntVect domain_hi(AMREX_D_DECL(params.size[0]-1,params.size[1]-1,params.size[2]-1));
        base_domain = Box(domain_lo, domain_hi);

        for (int n = 0; n < BL_SPACEDIM; n++) {
            real_box.setLo(n, 0.0);
            real_box.setHi(n, 3000.0);
        }
    } else if (params.ic_type == ICType::Census || params.ic_type == ICType::UrbanPop) {
        IntVect iv;
        iv[0] = iv[1] = (int) std::floor(std::sqrt((double)Ncommunities));
        while (iv[0]*iv[1] <= Ncommunities) {
            ++iv[0];
        }
        base_domain = Box(IntVect(AMREX_D_DECL(0, 0, 0)), iv-1);

        for (int n = 0; n < BL_SPACEDIM; n++) {
            real_box.setLo(n, 0.0);
            real_box.setHi(n, 1.0);
        }
    }

    geom.define(base_domain, &real_box, CoordSys::cartesian, is_per);
    return geom;
}
