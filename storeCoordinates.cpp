#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>

using Teuchos::RCP;
using SC = double;
using LO = Tpetra::Map<>::local_ordinal_type;
using GO = Tpetra::Map<>::global_ordinal_type;
using NO = Tpetra::Map<>::node_type;
using map_type = Tpetra::Map<LO, GO, NO>;
using mtx_type = Tpetra::CrsMatrix<SC, LO, GO, NO>;
using multiVect_type = Tpetra::MultiVector<SC, LO, GO, NO>;
using op_type = Tpetra::Operator<SC, LO, GO, NO>;

using namespace std;

// Structure to hold 3D coordinates
struct Point3D {
    double x;
    double y;
    double z;
};

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, nullptr);
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);

  {
    RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const size_t myRank = comm->getRank();
    int NumProc = comm->getSize();

    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& fancyout = *fancy;
    fancyout.setOutputToRootOnly(0);

    int numCoords = 10;
    // Create a vector of structures to hold 10 points
    vector<Point3D> points(numCoords);

    // Open the input file
    ifstream inputFile("pointDataInput.txt");

    // Check if the file opened successfully
    if (!inputFile.is_open()) {
      cerr << "Error opening file!" << endl;
      return 1;
    }

    // Read the coordinates from the file
    for (int i = 0; i < numCoords; ++i) {
      inputFile >> points[i].x >> points[i].y >> points[i].z;
    }

    // Close the input file
    inputFile.close();

    // Print the coordinates
    fancyout << "Coordinates of the points:" << endl;
    for (int i = 0; i < numCoords; ++i) {
      fancyout << "Point " << i + 1 << ": (" << points[i].x << ", " << points[i].y << ", " << points[i].z << ")" << endl;
    }

    // Create a Tpetra::MultiVector for coordinates
    RCP<multiVect_type> zoltanCoords = Teuchos::null;
    RCP<map_type> tpMap = rcp(new map_type(numCoords,0,comm));
    zoltanCoords = rcp(new multiVect_type(tpMap,3));

    int myNumElem = tpMap->getLocalNumElements();
    int lowGlobIndx = tpMap->getMinGlobalIndex();
    int highGlobIndx = tpMap->getMaxGlobalIndex();

    cout << "myRank: " << myRank << "; lowGlobIndx: " << lowGlobIndx << endl;
    cout << "myRank: " << myRank << "; highGlobIndx: " << highGlobIndx << endl;

    // Copy data from 'points' to the Tpetra MV
    for (int mtxRowIndex=lowGlobIndx; mtxRowIndex<=highGlobIndx; mtxRowIndex++) {
      zoltanCoords->replaceGlobalValue(mtxRowIndex,0,points[mtxRowIndex].x);
      zoltanCoords->replaceGlobalValue(mtxRowIndex,1,points[mtxRowIndex].y);
      zoltanCoords->replaceGlobalValue(mtxRowIndex,2,points[mtxRowIndex].z);
    }

    // This next block only prints back the values stored to the multivector
    // You can keep commented if you don't need the positive feedback.
    /*
    auto localView = zoltanCoords->getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t i = 0; i < localView.extent(0); ++i) {
      for (size_t j = 0; j < localView.extent(1); ++j) {
        std::cout << localView(i, j) << " ";
      }
      std::cout << std::endl;
    }*/

  }
  return 0;
}
