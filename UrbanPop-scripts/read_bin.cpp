#include <iostream>
#include <fstream>
#include <vector>

#include "UrbanPopAgentStruct.H"

using namespace std;

int main(int argc, char** argv) {
    ifstream f(argv[1], std::ios::binary);
    int lines = 0;
    vector<UrbanPopAgent> agents;
    for (;; lines++) {
        UrbanPopAgent agent;
        agent.read_binary(f);
        if (lines == 0) {
            // check first agent is set to -99 values to ensure formatting is correct
            agent.check_binary_inputs();
        } else {
            agents.push_back(agent);
        }
        if (f.peek() == EOF) break;
    }
    cout << agents.back() << endl;
    cout << "Read " << lines << " lines\n";
    return 0;
}
