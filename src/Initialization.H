/*! @file Initialization.H
    \brief Defines #ExaEpi::Initialization namespace
*/

#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include <vector>
#include <string>

#include "AgentContainer.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "Utils.H"

#include <AMReX_iMultiFab.H>

namespace ExaEpi
{
/*! \brief Namespace with initialization-related functions. */
namespace Initialization
{
    void read_workerflow (const DemographicData& demo, const TestParams& params, AgentContainer& pc);

    void assignTeachersAndWorkgroup (const DemographicData& demo, AgentContainer& pc);

    void setInitialCasesFromFile (AgentContainer& pc, const std::vector<CaseData> &cases, const std::vector<std::string>& names,
                                  const DemographicData& demo);

    void setInitialCasesRandom (AgentContainer& pc, std::vector<int> num_cases, const std::vector<std::string>& names,
                                const DemographicData& demo);
}
}

#endif
